# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import argparse
import dataclasses
import logging
import os
import pickle
import shutil
import sys
import types

import numpy as np
import torch
from ai_models.model import Model
from ai_models.sensitivity import add_sensitivity_parser_arguments
from ai_models.sensitivity import parse_target_area
from ai_models.sensitivity import SensitivityTarget
from ai_models.sensitivity import target_slug
from torch.utils.checkpoint import checkpoint

try:
    import timm.models.layers.helpers as _timm_helpers  # noqa: F401
except ModuleNotFoundError:
    try:
        from timm.layers.helpers import to_2tuple

        helpers_module = types.ModuleType("timm.models.layers.helpers")
        helpers_module.to_2tuple = to_2tuple
        sys.modules.setdefault("timm.models.layers.helpers", helpers_module)
    except Exception:
        pass


LOG = logging.getLogger(__name__)

try:
    from aurora import Batch
    from aurora import Metadata
    from aurora.model.aurora import Aurora
    from aurora.model.aurora import AuroraHighRes
    from huggingface_hub import hf_hub_download
except ModuleNotFoundError as e:
    msg = "You need microsoft-aurora and huggingface_hub installed to use this model."
    LOG.error(msg)
    raise ModuleNotFoundError(f"{msg}\n{e}")


class AuroraModel(Model):
    download_url = "https://huggingface.co/microsoft/aurora/resolve/main/{file}"

    area = [90, 0, -90, 360 - 0.25]
    grid = [0.25, 0.25]

    surf_vars = ("2t", "10u", "10v", "msl")
    atmos_vars = ("z", "u", "v", "t", "q")
    levels = (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)

    lagged = (-6, 0)

    param_sfc = surf_vars
    param_level_pl = (atmos_vars, levels)

    ordering = []

    expver = "auro"
    lora = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ordering = list(self.surf_vars) + [
            f"{param}{level}" for param in self.atmos_vars for level in self.levels
        ]
        self.level_to_index = {int(level): index for index, level in enumerate(self.levels)}
        self._sensitivity_latitudes = None
        self._sensitivity_longitudes = None
        self._warned_rollout_checkpointing = False

        default_sensitivity_path = f"{self.__class__.__name__.lower()}-sensitivity-{self.lead_time:03d}h.nc"
        self.init_sensitivity("aurora", default_sensitivity_path)

    def download_assets(self, **kwargs):
        del kwargs
        for filename in self.download_files:
            asset_path = os.path.realpath(os.path.join(self.assets, filename))
            if os.path.exists(asset_path):
                continue

            os.makedirs(os.path.dirname(asset_path), exist_ok=True)
            LOG.info("Downloading %s", asset_path)
            cache_path = hf_hub_download(repo_id="microsoft/aurora", filename=filename)
            shutil.copy2(cache_path, asset_path)

    def parse_model_args(self, args):
        parser = argparse.ArgumentParser(add_help=False)
        add_sensitivity_parser_arguments(parser)
        parser.add_argument(
            "--lora",
            type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
            nargs="?",
            const=True,
            default=None,
            help="Use LoRA model (true/false). Default depends on the model.",
        )
        parser.add_argument(
            "--model-checkpointing",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Enable Aurora activation checkpointing. Defaults to on in sensitivity mode.",
        )
        parser.add_argument(
            "--rollout-checkpointing",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Enable rollout checkpointing. Reserved for consistency with shared sensitivity options.",
        )
        return parser.parse_args(args)

    @property
    def latitudes(self):
        if self._sensitivity_latitudes is not None:
            return self._sensitivity_latitudes
        return np.asarray(self.fields_sfc[0].metadata("distinctLatitudes"), dtype=np.float32)

    @property
    def longitudes(self):
        if self._sensitivity_longitudes is not None:
            return self._sensitivity_longitudes
        values = np.asarray(self.fields_sfc[0].metadata("distinctLongitudes"), dtype=np.float32)
        return np.mod(values, 360.0)

    def parse_target_field(self, target_field=None, target_param=None, target_level=None):
        if target_field and target_param:
            raise ValueError("Use either --target-field or --target-param, not both")

        if target_field:
            if target_field not in self.ordering:
                raise ValueError(
                    f"Unknown target field '{target_field}'. Expected one of: {', '.join(self.ordering)}"
                )
            return target_field

        if target_param is None:
            if target_level is not None:
                raise ValueError("--target-level requires --target-param")
            return None

        if target_param in self.surf_vars:
            if target_level is not None:
                raise ValueError(f"Surface target '{target_param}' does not take --target-level")
            return target_param

        if target_param in self.atmos_vars:
            if target_level is None:
                raise ValueError(f"Pressure-level target '{target_param}' requires --target-level")
            if target_level not in self.levels:
                raise ValueError(
                    f"Unsupported level {target_level} for '{target_param}'. Expected one of: {self.levels}"
                )
            return f"{target_param}{target_level}"

        raise ValueError(f"Unsupported target parameter '{target_param}'")

    def default_target(self):
        field = self.parse_target_field(self.target_field, self.target_param, self.target_level)
        area = parse_target_area(self.target_area)
        if area is not None and field is None:
            raise ValueError("--target-area requires --target-field or --target-param")

        name = target_slug(field, "full-state")
        return SensitivityTarget(name=name, field=field, area=area, metric=self.sensitivity_metric)

    def config_targets(self, config):
        targets = config.get("targets")
        if not targets:
            return [self.default_target()]

        parsed_targets = []
        for index, target in enumerate(targets):
            if not isinstance(target, dict):
                raise ValueError("Each target entry must be a mapping")

            field = self.parse_target_field(
                target_field=target.get("field"),
                target_param=target.get("param"),
                target_level=target.get("level"),
            )
            area = parse_target_area(target.get("area"))
            if area is not None and field is None:
                raise ValueError("Target area requires a specific target field")

            metric = target.get("metric", self.sensitivity_metric)
            if metric not in ("mean", "mean-square"):
                raise ValueError(f"Unsupported metric '{metric}' for target {index + 1}")

            name = target_slug(target.get("name"), field or f"target-{index + 1}")
            parsed_targets.append(SensitivityTarget(name=name, field=field, area=area, metric=metric))

        return parsed_targets

    @staticmethod
    def _pressure_field_parts(field):
        for param in ("z", "u", "v", "t", "q"):
            if field.startswith(param):
                level_text = field[len(param) :]
                if level_text.isdigit():
                    return param, int(level_text)
                break
        raise ValueError(f"Could not parse pressure-level field '{field}'")

    def channel_to_variable_level(self, field):
        if field in self.surf_vars:
            return ("surf", field, None)

        param, level = self._pressure_field_parts(field)
        if param not in self.atmos_vars:
            raise ValueError(f"Unsupported atmospheric variable '{param}'")
        if level not in self.level_to_index:
            raise ValueError(f"Unsupported pressure level '{level}' for field '{field}'")
        return ("atmos", param, level)

    def load_model_instance(self, use_lora):
        LOG.info("Model is %s, use_lora=%s", self.__class__.__name__, use_lora)
        model = self.klass(use_lora=use_lora)
        model = model.to(self.device)

        checkpoint_path = os.path.join(self.assets, os.path.basename(self.checkpoint))
        if os.path.exists(checkpoint_path):
            LOG.info("Loading Aurora model from %s", checkpoint_path)
            model.load_checkpoint_local(checkpoint_path, strict=False)
        else:
            LOG.info("Downloading Aurora model %s", self.checkpoint)
            model.load_checkpoint("microsoft/aurora", self.checkpoint, strict=False)

        if self.model_checkpointing and hasattr(model, "configure_activation_checkpointing"):
            LOG.info("Model checkpointing enabled: configuring Aurora activation checkpointing")
            model.configure_activation_checkpointing()

        LOG.info("Loading Aurora model to device %s", self.device)
        model = model.to(self.device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        return model

    def load_static_vars(self):
        path = os.path.join(self.assets, self.download_files[0])
        if not os.path.exists(path):
            cache_path = hf_hub_download(repo_id="microsoft/aurora", filename=self.download_files[0])
            os.makedirs(os.path.dirname(path), exist_ok=True)
            shutil.copy2(cache_path, path)

        with open(path, "rb") as file_handle:
            static_vars = pickle.load(file_handle)

        return {name: torch.from_numpy(values.astype(np.float32)) for name, values in static_vars.items()}

    def build_input_batch(self):
        fields_pl = self.fields_pl
        fields_sfc = self.fields_sfc

        nj, ni = fields_pl[0].shape
        templates = {}
        surf_inputs = {}

        for param in self.surf_vars:
            field = fields_sfc.sel(param=param).order_by(valid_datetime="ascending")
            templates[param] = field[-1]
            values = field.to_numpy(dtype=np.float32)
            surf_inputs[param] = torch.from_numpy(values).unsqueeze(0)

        static_vars = self.load_static_vars()

        atmos_inputs = {}
        for param in self.atmos_vars:
            field = fields_pl.sel(param=param).order_by(valid_datetime="ascending", level=self.levels)
            for level in self.levels:
                templates[(param, level)] = field.sel(level=level)[-1]

            values = field.to_numpy(dtype=np.float32).reshape(len(self.lagged), len(self.levels), nj, ni)
            atmos_inputs[param] = torch.from_numpy(values).unsqueeze(0)

        north, west, south, east = self.area
        metadata = Metadata(
            lat=torch.linspace(north, south, nj),
            lon=torch.linspace(west, east, ni),
            time=(self.start_datetime,),
            atmos_levels=self.levels,
        )

        batch = Batch(
            surf_vars=surf_inputs,
            static_vars=static_vars,
            atmos_vars=atmos_inputs,
            metadata=metadata,
        )

        return batch, templates, nj, ni

    def nan_extend(self, data):
        return np.concatenate(
            (data, np.full_like(data[[-1], :], np.nan, dtype=data.dtype)),
            axis=0,
        )

    def write_prediction_batch(self, prediction, templates, step, nj, ni):
        for param, values in prediction.surf_vars.items():
            data = np.squeeze(values.detach().cpu().numpy())
            data = self.nan_extend(data)
            assert data.shape == (nj, ni)
            self.write(data, template=templates[param], step=step, check_nans=True)

        for param, values in prediction.atmos_vars.items():
            tensor = values.detach().cpu().numpy()
            for level_index, level in enumerate(self.levels):
                data = np.squeeze(tensor[:, :, level_index])
                data = self.nan_extend(data)
                assert data.shape == (nj, ni)
                self.write(data, template=templates[(param, level)], step=step, check_nans=True)

    def forecast_step_count(self):
        if self.lead_time <= 0:
            raise ValueError(f"lead_time must be positive, got {self.lead_time}")
        if self.lead_time % 6 != 0:
            raise ValueError(f"For Aurora, lead_time must be a multiple of 6 hours; got {self.lead_time}")
        return self.lead_time // 6

    @staticmethod
    def advance_rollout(current_batch, prediction):
        device = next(iter(prediction.surf_vars.values())).device
        current_batch = current_batch.to(device)
        return dataclasses.replace(
            prediction,
            surf_vars={
                name: torch.cat([current_batch.surf_vars[name][:, 1:], values], dim=1)
                for name, values in prediction.surf_vars.items()
            },
            atmos_vars={
                name: torch.cat([current_batch.atmos_vars[name][:, 1:], values], dim=1)
                for name, values in prediction.atmos_vars.items()
            },
        )

    def model_step(self, model, batch):
        if self.rollout_checkpointing and torch.is_grad_enabled():
            if not self._warned_rollout_checkpointing:
                LOG.info("Rollout checkpointing enabled: rematerializing each Aurora rollout step")
                self._warned_rollout_checkpointing = True

            def _forward(batch_value):
                return model(batch_value)

            return checkpoint(_forward, batch, use_reentrant=False)

        return model(batch)

    @staticmethod
    def _target_weights(latitudes, longitudes, area, *, device, dtype):
        latitudes = torch.as_tensor(latitudes, dtype=dtype, device=device)
        longitudes = torch.as_tensor(np.mod(longitudes, 360.0), dtype=dtype, device=device)

        lat_weights = torch.cos(torch.deg2rad(latitudes)).clamp_min(0)
        lat_mask = torch.ones_like(lat_weights, dtype=torch.bool)
        lon_mask = torch.ones_like(longitudes, dtype=torch.bool)

        if area is not None:
            north, west, south, east = area
            lat_mask = (latitudes <= north) & (latitudes >= south)
            if west <= east:
                lon_mask = (longitudes >= west) & (longitudes <= east)
            else:
                lon_mask = (longitudes >= west) | (longitudes <= east)

        weights = lat_weights[:, None] * lat_mask.to(dtype)[:, None] * lon_mask.to(dtype)[None, :]
        total = weights.sum()
        if total <= 0:
            raise ValueError("Selected target area does not overlap the Aurora grid")
        return weights / total

    def _target_tensor(self, prediction, field):
        kind, param, level = self.channel_to_variable_level(field)
        if kind == "surf":
            return prediction.surf_vars[param][0, -1]

        level_index = self.level_to_index[int(level)]
        return prediction.atmos_vars[param][0, -1, level_index]

    def sensitivity_objective(self, prediction, target):
        metric = target.metric
        if metric not in ("mean", "mean-square"):
            raise ValueError(f"Unsupported sensitivity metric: {metric}")

        if target.field is None:
            total_sum = torch.zeros((), dtype=torch.float32, device=self.device)
            total_count = 0

            for values in prediction.surf_vars.values():
                field_values = values[:, -1]
                if metric == "mean":
                    total_sum = total_sum + field_values.sum()
                else:
                    total_sum = total_sum + field_values.square().sum()
                total_count += int(field_values.numel())

            for values in prediction.atmos_vars.values():
                field_values = values[:, -1]
                if metric == "mean":
                    total_sum = total_sum + field_values.sum()
                else:
                    total_sum = total_sum + field_values.square().sum()
                total_count += int(field_values.numel())

            if total_count <= 0:
                raise ValueError("Aurora output contained no values for sensitivity objective")
            return total_sum / float(total_count)

        field_values = self._target_tensor(prediction, target.field)
        if target.area is None:
            if metric == "mean":
                return field_values.mean()
            return field_values.square().mean()

        latitudes = prediction.metadata.lat.detach().cpu().numpy().astype(np.float32)
        longitudes = prediction.metadata.lon.detach().cpu().numpy().astype(np.float32)
        weights = self._target_weights(
            latitudes,
            longitudes,
            target.area,
            device=field_values.device,
            dtype=field_values.dtype,
        )
        if metric == "mean":
            return (field_values * weights).sum()
        return (field_values.square() * weights).sum()

    def gradients_to_channel_maps(self, gradients, lag_zero_index):
        channel_maps = []
        surf_count = len(self.surf_vars)

        for index in range(surf_count):
            gradient = gradients[index]
            channel_maps.append(gradient[0, lag_zero_index].detach().cpu().numpy().astype(np.float32))

        for index in range(len(self.atmos_vars)):
            gradient = gradients[surf_count + index]
            for level_index in range(len(self.levels)):
                channel_maps.append(
                    gradient[0, lag_zero_index, level_index].detach().cpu().numpy().astype(np.float32)
                )

        return np.stack(channel_maps, axis=0)

    def run_forecast(self, model, batch, templates, nj, ni):
        forecast_steps = self.forecast_step_count()
        current_batch = batch.crop(model.patch_size)

        with torch.inference_mode():
            with self.stepper(6) as stepper:
                for index in range(forecast_steps):
                    prediction = self.model_step(model, current_batch)
                    step = (index + 1) * 6
                    self.write_prediction_batch(prediction, templates, step, nj, ni)
                    current_batch = self.advance_rollout(current_batch, prediction)
                    stepper(index, step)

    def run_sensitivity(self, model, batch, templates, nj, ni):
        forecast_steps = self.forecast_step_count()
        lag_zero_index = list(self.lagged).index(0) if 0 in self.lagged else len(self.lagged) - 1

        surface_inputs = {
            name: tensor.to(self.device).detach().clone().requires_grad_(True)
            for name, tensor in batch.surf_vars.items()
        }
        atmospheric_inputs = {
            name: tensor.to(self.device).detach().clone().requires_grad_(True)
            for name, tensor in batch.atmos_vars.items()
        }
        static_inputs = {name: tensor.to(self.device) for name, tensor in batch.static_vars.items()}

        metadata = Metadata(
            lat=batch.metadata.lat.to(self.device),
            lon=batch.metadata.lon.to(self.device),
            time=batch.metadata.time,
            atmos_levels=batch.metadata.atmos_levels,
            rollout_step=batch.metadata.rollout_step,
        )

        current_batch = Batch(
            surf_vars=surface_inputs,
            static_vars=static_inputs,
            atmos_vars=atmospheric_inputs,
            metadata=metadata,
        )
        current_batch = current_batch.crop(model.patch_size)

        self._sensitivity_latitudes = metadata.lat.detach().cpu().numpy().astype(np.float32)
        self._sensitivity_longitudes = np.mod(metadata.lon.detach().cpu().numpy().astype(np.float32), 360.0)

        objective_batch = None
        with self.stepper(6) as stepper:
            for index in range(forecast_steps):
                prediction = self.model_step(model, current_batch)
                step = (index + 1) * 6
                self.write_prediction_batch(prediction, templates, step, nj, ni)
                objective_batch = prediction
                current_batch = self.advance_rollout(current_batch, prediction)
                stepper(index, step)

        if objective_batch is None:
            raise ValueError("Sensitivity rollout produced no prediction steps")

        objectives = []
        for target in self.sensitivity_manager.targets:
            self.sensitivity_manager.current_target = target
            objectives.append(self.sensitivity_manager.objective(objective_batch, target))

        input_tensors = [surface_inputs[name] for name in self.surf_vars] + [
            atmospheric_inputs[name] for name in self.atmos_vars
        ]

        gradients = []
        objective_values = []
        for index, objective in enumerate(objectives):
            gradient_tensors = torch.autograd.grad(
                objective,
                input_tensors,
                retain_graph=(index < len(objectives) - 1),
            )
            gradients.append(self.gradients_to_channel_maps(gradient_tensors, lag_zero_index))
            objective_values.append(float(objective.detach().cpu()))

        stacked_gradients = np.stack(gradients, axis=0)
        self.sensitivity_manager.save(stacked_gradients, objective_values)

    def run(self):
        if self.sensitivity_config and not self.sensitivity:
            self.sensitivity = True

        use_lora = self.lora if self.lora is not None else self.use_lora
        model = self.load_model_instance(use_lora=use_lora)

        self.write_input_fields(self.fields_pl + self.fields_sfc)
        batch, templates, nj, ni = self.build_input_batch()

        LOG.info("Starting inference")
        if self.sensitivity:
            self.run_sensitivity(model, batch, templates, nj, ni)
            return

        self.run_forecast(model, batch, templates, nj, ni)


class Aurora0p25(AuroraModel):
    expver = "au25"

    klass = Aurora
    download_files = ("aurora-0.25-static.pickle",)
    area = [90, 0, -90, 360 - 0.25]
    grid = [0.25, 0.25]


class Aurora0p25Pretrained(Aurora0p25):
    use_lora = False
    checkpoint = "aurora-0.25-pretrained.ckpt"


class Aurora0p25FineTuned(Aurora0p25):
    use_lora = True
    checkpoint = "aurora-0.25-finetuned.ckpt"

    def patch_retrieve_request(self, request):
        if request.get("class", "od") != "od":
            return

        if request.get("type", "an") not in ("an", "fc"):
            return

        if request.get("stream", "oper") not in ("oper", "scda"):
            return

        request["type"] = "fc"

        time = request.get("time", 12)
        request["stream"] = {
            0: "oper",
            6: "scda",
            12: "oper",
            18: "scda",
        }[time]


class Aurora0p1FineTuned(AuroraModel):
    download_files = ("aurora-0.1-static.pickle",)
    area = [90, 0, -90, 360 - 0.1]
    grid = [0.1, 0.1]

    klass = AuroraHighRes
    use_lora = True
    checkpoint = "aurora-0.1-finetuned.ckpt"


def model(model_version, **kwargs):
    models = {
        "0.25-pretrained": Aurora0p25Pretrained,
        "0.25-finetuned": Aurora0p25FineTuned,
        "0.1-finetuned": Aurora0p1FineTuned,
        "default": Aurora0p1FineTuned,
        "latest": Aurora0p1FineTuned,
    }

    if model_version not in models:
        LOG.error("Model version %s not found, using default", model_version)
        LOG.error("Available models: %s", list(models.keys()))
        raise ValueError(f"Model version {model_version} not found")

    return models[model_version](**kwargs)
