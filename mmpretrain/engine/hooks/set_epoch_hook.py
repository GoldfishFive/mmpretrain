
from mmengine.hooks import Hook

from mmpretrain.registry import HOOKS
# from mmselfsup.utils import get_model
from mmengine.logging import MMLogger
from mmpretrain.utils import get_ori_model
@HOOKS.register_module()
class SetEpochHook(Hook):
    def __init__(
        self,
        start_epoch: int = 0,
    ) -> None:
        self.start_epoch = start_epoch

    def before_train_epoch(self, runner) -> None:
        # set the cur epoch to bacakbone
        get_ori_model(runner.model).backbone.set_epoch(runner.epoch)
        MMLogger.get_current_instance().info(
            f'cur_epoch is : {runner.epoch}')