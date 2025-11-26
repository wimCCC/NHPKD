# mmrazor/models/task_modules/deliveries/cross_stage_delivery.py
import functools

from mmrazor.registry import TASK_UTILS
from .method_outputs_delivery import MethodOutputsDelivery

@TASK_UTILS.register_module()
class CrossStageDelivery(MethodOutputsDelivery):
    def __init__(self, method_path, max_keep_data=1, alpha=0.5):
        super().__init__(method_path, max_keep_data)
        self.alpha = alpha  # 教师特征融合权重

    def deliver_wrapper(self, origin_method):
        @functools.wraps(origin_method)
        def wrapped(self, stage1_feat, stage2_feat, *args, **kwargs):
            if self.override_data:
                # 学生模式：混合教师特征
                tea_fused = self.data_queue.popleft()
                stu_fused = origin_method(self, stage1_feat, stage2_feat)
                return self.alpha * tea_fused + (1-self.alpha) * stu_fused
            else:
                # 教师模式：记录特征
                outputs = origin_method(self, stage1_feat, stage2_feat)
                self.data_queue.append(outputs)
                return outputs
        return wrapped