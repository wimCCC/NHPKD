from mmrazor.registry import MODELS
from .configurable_distiller import ConfigurableDistiller
from typing import Dict, Optional, Union, List
from ..task_modules import DistillDeliveryManager, RecorderManager
from ..algorithms.base import LossResults
from torch import nn
from mmengine.model import BaseModel
from inspect import signature
@MODELS.register_module()
class FuseDistiller(ConfigurableDistiller):
    """FuseDistiller is a distiller that can fuse multiple losses into one loss.

    Args:
        loss_weights (dict): The weight of each loss.
    """

    def __init__(self,
                 student_recorders: Optional[Dict[str, Dict]] = None,
                 teacher_recorders: Optional[Dict[str, Dict]] = None,
                 distill_deliveries: Optional[Dict[str, Dict]] = None,
                 connectors: Optional[Dict[str, Dict]] = None,
                 distill_losses: Optional[Dict[str, Dict]] = None,
                 loss_forward_mappings: Optional[Dict[str, Dict]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.student_recorders = RecorderManager(student_recorders)
        self.teacher_recorders = RecorderManager(teacher_recorders)

        self.deliveries = DistillDeliveryManager(distill_deliveries)

        self.distill_losses = self.build_distill_losses(distill_losses)

        self.connectors = self.build_connectors(connectors)

        if loss_forward_mappings:
            # Check if loss_forward_mappings is in the correct format.
            self._check_loss_forward_mappings(self.distill_losses,
                                              loss_forward_mappings,
                                              self.student_recorders,
                                              self.teacher_recorders)
            self.loss_forward_mappings = loss_forward_mappings
        else:
            self.loss_forward_mappings = dict()

    def set_deliveries_override(self, override: bool) -> None:
        """Set the `override_data` of all deliveries."""
        self.deliveries.override_data = override

    def set_deliveries_override(self, override: bool) -> None:
        """Set the `override_data` of all deliveries."""
        self.deliveries.override_data = override

    def prepare_from_student(self, model: BaseModel) -> None:
        """Initialize student recorders."""
        self.student_recorders.initialize(model)

    def prepare_from_teacher(self, model: nn.Module) -> None:
        """Initialize teacher recorders."""
        self.teacher_recorders.initialize(model)

    def build_connectors(
        self,
        connectors: Optional[Union[Dict[str, List], Dict[str, Dict]]] = None,
    ) -> nn.ModuleDict:
        """Initialize connectors."""

        distill_connecotrs = nn.ModuleDict()
        if connectors:
            for connector_name, connector_cfg in connectors.items():
                if isinstance(connector_cfg, dict):
                    connector = MODELS.build(connector_cfg)
                    distill_connecotrs[connector_name] = connector
                else:
                    assert isinstance(connector_cfg, list)
                    module_list = []
                    for cfg in connector_cfg:
                        connector = MODELS.build(cfg)
                        module_list.append(connector)
                    distill_connecotrs[connector_name] = nn.Sequential(
                        *module_list)

        return distill_connecotrs

    def build_distill_losses(
        self,
        losses: Optional[Dict[str, Dict]] = None,
    ) -> nn.ModuleDict:
        """build distill losses according config."""

        distill_losses = nn.ModuleDict()
        if losses:
            for loss_name, loss_cfg in losses.items():
                assert loss_name not in distill_losses
                if 'loss' not in loss_name:
                    warnings.warn(
                        f'Warning: If {loss_name} is a loss that needs to '
                        f'backward, the name of {loss_name} must contain '
                        f'"loss". If it is only used as a statistical value, '
                        'then the name must not contain "loss". More details '
                        'see docs for '
                        ':func:`mmengine.model.BaseModel._parse_loss`',
                        UserWarning)
                item_loss = MODELS.build(loss_cfg)
                distill_losses[loss_name] = item_loss

        return distill_losses

    def get_record(self,
                recorder: Union[str, List[str]],
                from_student: bool ,
                record_idx: int = 0,
                connector: Optional[str] = None,
                data_idx: Optional[int] = None,
                connector_idx: Optional[int] = None) -> List:
        if isinstance(recorder, list):
            records = []
            for rec in recorder:
                recorder_ = self.student_recorders.get_recorder(rec) if from_student \
                        else self.teacher_recorders.get_recorder(rec)
                records.append(recorder_.get_record_data(record_idx, data_idx))
            record_data = records  # 返回列表形式的多组数据
        else:
        # 原有单记录器逻辑
            recorder_ = self.student_recorders.get_recorder(recorder) if from_student \
                    else self.teacher_recorders.get_recorder(recorder)
            record_data = recorder_.get_record_data(record_idx, data_idx)
        if connector:
            record_data = self.connectors[connector](record_data)
        if connector_idx is not None:
            record_data = record_data[connector_idx]

        return record_data
    def compute_distill_losses(self) -> LossResults:
        """Compute distill losses automatically."""
        # Record all computed losses' results.
        losses = dict()
        for loss_name, forward_mappings in self.loss_forward_mappings.items():
            forward_kwargs = dict()
            for forward_key, record in forward_mappings.items():
                forward_var = self.get_record(**record)
                forward_kwargs[forward_key] = forward_var

            loss_module = self.distill_losses[loss_name]
            loss = loss_module(**forward_kwargs)  # type: ignore
            # add computed loss result.
            losses[loss_name] = loss
        return losses
    def _check_loss_forward_mappings(
        self, losses: nn.ModuleDict, loss_forward_mappings: Dict[str,
                                                                    Dict],
        student_recorders: RecorderManager,
        teacher_recorders: RecorderManager) -> None:

        if not isinstance(loss_forward_mappings, dict):
            raise TypeError(
                'loss_forward_mappings should be a dict instance, but got'
                f'{type(loss_forward_mappings)}')

        for loss_name, forward_mappings in loss_forward_mappings.items():
            assert loss_name in losses, \
                f'"{loss_name}" is not in distill losses. The keys of ' \
                'loss_forward_kwargs must match the keys of distill_losses.'

            if not isinstance(forward_mappings, dict):
                raise TypeError(
                    'Each item of loss_forward_mappings should be a dict '
                    f'instance, but got {type(forward_mappings)}')

            loss_module = losses[loss_name]
            loss_forward_params = signature(loss_module.forward).parameters
            loss_forward_keys = loss_forward_params.keys()
            # Allow default params.
            # Check non-default params, not len(params).

            for forward_key, record_info in forward_mappings.items():
                assert forward_key in loss_forward_keys, \
                    f'{forward_key} is not in the signature of \
                    {type(loss_module).__name__} forward, \
                    please check your config.'

                if (loss_forward_params[forward_key].default !=
                        loss_forward_params[forward_key].empty):
                    # default params without check
                    continue

                assert 'recorder' in record_info, \
                    'Each item of loss_forward_mappings should have ' \
                    '"recorder", pls check your config.'

                assert 'from_student' in record_info, \
                    'Each item of loss_forward_mappings should have ' \
                    '"from_student", pls check your config.'

                recorder: str = record_info['recorder']
                from_student: bool = record_info['from_student']

                if not isinstance(from_student, bool):
                    raise TypeError(f'from_student should be a bool instance, '
                                    f'but got {type(from_student)}')

                if from_student:
                    assert recorder in student_recorders.recorders, \
                        f'For {forward_key}, "{recorder}" must be in \
                        `student_recorders`.'

                else:
                    assert recorder in teacher_recorders.recorders, \
                        f'For {forward_key}, "{recorder}" must be in \
                        `teacher_recorders`.'

                if 'connector' in record_info:
                    connector: str = record_info['connector']
                    assert connector in self.connectors, \
                        f'{connector} must be in "connectors".'

        