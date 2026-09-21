# 鑷畾涔夎繍琛屾椂閰嶇疆

## 鑷畾涔変紭鍖栧櫒璁剧疆

浼樺寲鍣ㄧ浉鍏崇殑閰嶇疆鏄敱 `optim_wrapper` 绠＄悊鐨勶紝鍏堕€氬父鏈変笁涓瓧娈碉細`optimizer`锛宍paramwise_cfg`锛宍clip_grad`銆傛洿澶氱粏鑺傝鍙傝€?[OptimWrapper](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/optim_wrapper.html)銆傚涓嬫墍绀猴紝浣跨敤 `AdamW` 浣滀负`浼樺寲鍣╜锛岄骞茬綉缁滅殑瀛︿範鐜囬檷浣?10 鍊嶏紝骞舵坊鍔犱簡姊害瑁佸壀銆?
```python
optim_wrapper = dict(
    type='OptimWrapper',
    # 浼樺寲鍣?    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),

    # 鍙傛暟绾у涔犵巼鍙婃潈閲嶈“鍑忕郴鏁拌缃?    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
        },
        norm_decay_mult=0.0),

    # 姊害瑁佸壀
    clip_grad=dict(max_norm=0.01, norm_type=2))
```

### 鑷畾涔?PyTorch 鏀寔鐨勪紭鍖栧櫒

鎴戜滑宸茬粡鏀寔浣跨敤鎵€鏈?PyTorch 瀹炵幇鐨勪紭鍖栧櫒锛屼笖鍞竴闇€瑕佷慨鏀圭殑鍦版柟灏辨槸鏀瑰彉閰嶇疆鏂囦欢涓殑 `optim_wrapper` 瀛楁涓殑 `optimizer` 瀛楁銆備緥濡傦紝濡傛灉鎮ㄦ兂浣跨敤 `Adam`锛堟敞鎰忚繖鏍峰彲鑳戒細浣挎€ц兘澶у箙涓嬮檷锛夛紝鎮ㄥ彲浠ヨ繖鏍蜂慨鏀癸細

```python
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=0.0003, weight_decay=0.0001))
```

涓轰簡淇敼妯″瀷鐨勫涔犵巼锛岀敤鎴峰彧闇€瑕佷慨鏀?`optimizer` 涓殑 `lr` 瀛楁銆傜敤鎴峰彲浠ユ牴鎹?PyTorch 鐨?[API 鏂囨。](https://pytorch.org/docs/stable/optim.html?highlight=optim#module-torch.optim)鐩存帴璁剧疆鍙傛暟銆?
### 鑷畾涔夊苟瀹炵幇浼樺寲鍣?
#### 1. 瀹氫箟鏂扮殑浼樺寲鍣?
涓€涓嚜瀹氫箟浼樺寲鍣ㄥ彲浠ユ寜鐓у涓嬭繃绋嬪畾涔夛細

鍋囪鎮ㄦ兂瑕佹坊鍔犱竴涓彨 `MyOptimizer` 鐨勶紝鎷ユ湁鍙傛暟 `a`锛宍b` 鍜?`c` 鐨勪紭鍖栧櫒锛屾偍闇€瑕佸垱寤轰竴涓彨鍋?`mmdet3d/engine/optimizers` 鐨勭洰褰曘€傛帴涓嬫潵锛屽簲璇ュ湪鐩綍涓嬫煇涓枃浠朵腑瀹炵幇鏂扮殑浼樺寲鍣紝姣斿 `mmdet3d/engine/optimizers/my_optimizer.py`锛?
```python
from torch.optim import Optimizer

from mmdet3d.registry import OPTIMIZERS


@OPTIMIZERS.register_module()
class MyOptimizer(Optimizer):

    def __init__(self, a, b, c):
        pass
```

#### 2. 灏嗕紭鍖栧櫒娣诲姞鍒版敞鍐屽櫒

涓轰簡鎵惧埌涓婅堪瀹氫箟鐨勪紭鍖栧櫒妯″潡锛岃妯″潡棣栧厛闇€瑕佽寮曞叆涓诲懡鍚嶇┖闂淬€傛湁涓ょ瀹炵幇鏂规硶锛?
- 淇敼 `mmdet3d/engine/optimizers/__init__.py` 瀵煎叆璇ユā鍧椼€?
  鏂板畾涔夌殑妯″潡搴旇鍦?`mmdet3d/engine/optimizers/__init__.py` 涓瀵煎叆锛屼粠鑰岃鎵惧埌骞朵笖琚坊鍔犲埌娉ㄥ唽鍣ㄤ腑锛?
  ```python
  from .my_optimizer import MyOptimizer
  ```

- 鍦ㄩ厤缃腑浣跨敤 `custom_imports` 鏉ヤ汉宸ュ鍏ユ柊浼樺寲鍣ㄣ€?
  ```python
  custom_imports = dict(imports=['mmdet3d.engine.optimizers.my_optimizer'], allow_failed_imports=False)
  ```

  妯″潡 `mmdet3d.engine.optimizers.my_optimizer` 浼氬湪绋嬪簭寮€濮嬭瀵煎叆锛屼笖 `MyOptimizer` 绫诲湪閭ｆ椂浼氳嚜鍔ㄨ娉ㄥ唽銆傛敞鎰忓埌搴旇鍙湁鍖呭惈 `MyOptimizer` 绫荤殑鍖呰瀵煎叆銆俙mmdet3d.engine.optimizers.my_optimizer.MyOptimizer`**涓嶈兘**琚洿鎺ュ鍏ャ€?
  浜嬪疄涓婏紝鐢ㄦ埛鍙互鍦ㄨ繖绉嶅鍏ョ殑鏂规硶涓娇鐢ㄥ畬鍏ㄤ笉鍚岀殑鏂囦欢鐩綍缁撴瀯锛屽彧瑕佷繚璇佹牴鐩綍鑳藉湪 `PYTHONPATH` 涓瀹氫綅銆?
#### 3. 鍦ㄩ厤缃枃浠朵腑鎸囧畾浼樺寲鍣?
鎺ヤ笅鏉ユ偍鍙互鍦ㄩ厤缃枃浠剁殑 `optimizer` 瀛楁涓娇鐢?`MyOptimizer`銆傚湪閰嶇疆鏂囦欢涓紝浼樺寲鍣ㄥ湪 `optimizer` 瀛楁涓互濡備笅鏂瑰紡瀹氫箟锛?
```python
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))
```

涓轰簡浣跨敤鎮ㄨ嚜宸辩殑浼樺寲鍣紝璇ュ瓧娈靛彲浠ユ敼涓猴細

```python
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='MyOptimizer', a=a_value, b=b_value, c=c_value))
```

### 鑷畾涔変紭鍖栧櫒灏佽鏋勯€犲櫒

閮ㄥ垎妯″瀷鍙兘浼氭嫢鏈変竴浜涘弬鏁颁笓灞炵殑浼樺寲鍣ㄨ缃紝姣斿 BatchNorm 灞傜殑鏉冮噸琛板噺 (weight decay)銆傜敤鎴峰彲浠ラ€氳繃鑷畾涔変紭鍖栧櫒灏佽鏋勯€犲櫒鏉ュ閭ｄ簺缁嗙矑搴︾殑鍙傛暟杩涜璋冧紭銆?
```python
from mmengine.optim import DefaultOptimWrapperConstructor

from mmdet3d.registry import OPTIM_WRAPPER_CONSTRUCTORS
from .my_optimizer import MyOptimizer


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class MyOptimizerWrapperConstructor(DefaultOptimWrapperConstructor):

    def __init__(self,
                 optim_wrapper_cfg: dict,
                 paramwise_cfg: Optional[dict] = None):
        pass

    def __call__(self, model: nn.Module) -> OptimWrapper:

        return optim_wrapper
```

榛樿浼樺寲鍣ㄥ皝瑁呮瀯閫犲櫒鍦╗杩欓噷](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/optimizer/default_constructor.py#L18)瀹炵幇銆傝繖閮ㄥ垎浠ｇ爜涔熷彲浠ョ敤浣滄柊浼樺寲鍣ㄥ皝瑁呮瀯閫犲櫒鐨勬ā鏉裤€?
### 棰濆鐨勮缃?
娌℃湁鍦ㄤ紭鍖栧櫒閮ㄥ垎瀹炵幇鐨勬妧宸у簲璇ラ€氳繃浼樺寲鍣ㄥ皝瑁呮瀯閫犲櫒鎴栬€呴挬瀛愭潵瀹炵幇锛堟瘮濡傞€愬弬鏁扮殑瀛︿範鐜囪缃級銆傛垜浠垪涓句簡涓€浜涘父鐢ㄧ殑鍙互绋冲畾璁粌杩囩▼鎴栬€呭姞閫熻缁冪殑璁剧疆銆傛垜浠杩庢彁渚涙洿澶氱被浼艰缃殑 PR 鍜?issue銆?
- __浣跨敤姊害瑁佸壀 (gradient clip) 鏉ョǔ瀹氳缁冭繃绋媉_锛氫竴浜涙ā鍨嬩緷璧栨搴﹁鍓妧鏈潵瑁佸壀璁粌涓殑姊害锛屼互绋冲畾璁粌杩囩▼銆備妇渚嬪涓嬶細

  ```python
  optim_wrapper = dict(
      _delete_=True, clip_grad=dict(max_norm=35, norm_type=2))
  ```

  濡傛灉鎮ㄧ殑閰嶇疆缁ф壙浜嗕竴涓凡缁忚缃簡 `optim_wrapper` 鐨勫熀纭€閰嶇疆锛岄偅涔堟偍鍙兘闇€瑕?`_delete_=True` 瀛楁鏉ヨ鐩栧熀纭€閰嶇疆涓棤鐢ㄧ殑璁剧疆銆傛洿澶氱粏鑺傝鍙傝€僛閰嶇疆鏂囨。](https://mmdetection3d.readthedocs.io/zh_CN/dev-1.x/user_guides/config.html)銆?
- __浣跨敤鍔ㄩ噺璋冨害鍣?(momentum scheduler) 鏉ュ姞閫熸ā鍨嬫敹鏁沖_锛氭垜浠敮鎸佺敤鍔ㄩ噺璋冨害鍣ㄦ潵鏍规嵁瀛︿範鐜囨洿鏀规ā鍨嬬殑鍔ㄩ噺锛岃繖鏍峰彲浠ヤ娇妯″瀷鏇村揩鍦版敹鏁涖€傚姩閲忚皟搴﹀櫒閫氬父鍜屽涔犵巼璋冨害鍣ㄤ竴璧蜂娇鐢紝渚嬪锛屽涓嬮厤缃枃浠跺湪 [3D 妫€娴媇(https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/configs/_base_/schedules/cyclic-20e.py)涓鐢ㄤ簬鍔犻€熸ā鍨嬫敹鏁涖€傛洿澶氱粏鑺傝鍙傝€?[CosineAnnealingLR](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/scheduler/lr_scheduler.py#L43) 鍜?[CosineAnnealingMomentum](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/scheduler/momentum_scheduler.py#L71) 鐨勫疄鐜版柟娉曘€?
  ```python
  param_scheduler = [
      # 瀛︿範鐜囪皟搴﹀櫒
      # 鍦ㄥ墠 8 涓?epoch锛屽涔犵巼浠?0 鍗囧埌 lr * 10
      # 鍦ㄦ帴涓嬫潵 12 涓?epoch锛屽涔犵巼浠?lr * 10 闄嶅埌 lr * 1e-4
      dict(
          type='CosineAnnealingLR',
          T_max=8,
          eta_min=lr * 10,
          begin=0,
          end=8,
          by_epoch=True,
          convert_to_iter_based=True),
      dict(
          type='CosineAnnealingLR',
          T_max=12,
          eta_min=lr * 1e-4,
          begin=8,
          end=20,
          by_epoch=True,
          convert_to_iter_based=True),
      # 鍔ㄩ噺璋冨害鍣?      # 鍦ㄥ墠 8 涓?epoch锛屽姩閲忎粠 0 鍗囧埌 0.85 / 0.95
      # 鍦ㄦ帴涓嬫潵 12 涓?epoch锛屽姩閲忎粠 0.85 / 0.95 鍗囧埌 1
      dict(
          type='CosineAnnealingMomentum',
          T_max=8,
          eta_min=0.85 / 0.95,
          begin=0,
          end=8,
          by_epoch=True,
          convert_to_iter_based=True),
      dict(
          type='CosineAnnealingMomentum',
          T_max=12,
          eta_min=1,
          begin=8,
          end=20,
          by_epoch=True,
          convert_to_iter_based=True)
  ]
  ```

## 鑷畾涔夎缁冭皟搴?
榛樿鎯呭喌涓嬫垜浠娇鐢ㄩ樁姊紡瀛︿範鐜囪“鍑忕殑 1 鍊嶈缁冭皟搴︼紝杩欎細璋冪敤 MMEngine 涓殑 [`MultiStepLR`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/scheduler/lr_scheduler.py#L144)銆傛垜浠湪[杩欓噷](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/scheduler/lr_scheduler.py)鏀寔浜嗗緢澶氬叾浠栧涔犵巼璋冨害锛屾瘮濡俙浣欏鸡閫€鐏玚鍜宍澶氶」寮忚“鍑廯璋冨害銆備笅闈㈡槸涓€浜涙牱渚嬶細

- 澶氶」寮忚“鍑忚皟搴︼細

  ```python
  param_scheduler = [
      dict(
          type='PolyLR',
          power=0.9,
          eta_min=1e-4,
          begin=0,
          end=8,
          by_epoch=True)]
  ```

- 浣欏鸡閫€鐏皟搴︼細

  ```python
  param_scheduler = [
      dict(
          type='CosineAnnealingLR',
          T_max=8,
          eta_min=lr * 1e-5,
          begin=0,
          end=8,
          by_epoch=True)]
  ```

## 鑷畾涔夎缁冨惊鐜帶鍒跺櫒

榛樿鎯呭喌涓嬶紝鎴戜滑鍦?`train_cfg` 涓娇鐢?`EpochBasedTrainLoop`锛屽苟鍦ㄦ瘡涓€涓缁?epoch 瀹屾垚鍚庤繘琛屼竴娆￠獙璇侊紝濡備笅鎵€绀猴細

```python
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_begin=1, val_interval=1)
```

浜嬪疄涓婏紝[`IterBasedTrainLoop`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py#L185) 鍜?[`EpochBasedTrainLoop`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py#L18) 閮芥敮鎸佸姩鎬侀棿闅旈獙璇侊紝濡備笅鎵€绀猴細

```python
# 鍦ㄧ 365001 娆¤凯浠ｄ箣鍓嶏紝鎴戜滑姣忛殧 5000 娆¤凯浠ｉ獙璇佷竴娆°€?# 鍦ㄧ 365000 娆¤凯浠ｄ箣鍚庯紝鎴戜滑姣忛殧 368750 娆¤凯浠ｉ獙璇佷竴娆★紝
# 杩欐剰鍛崇潃鎴戜滑鍦ㄨ缁冪粨鏉熷悗杩涜楠岃瘉銆?
interval = 5000
max_iters = 368750
dynamic_intervals = [(max_iters // interval * interval + 1, max_iters)]
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=interval,
    dynamic_intervals=dynamic_intervals)
```

## 鑷畾涔夐挬瀛?
### 鑷畾涔夊苟瀹炵幇閽╁瓙

#### 1. 瀹炵幇涓€涓柊閽╁瓙

MMEngine 鎻愪緵浜嗕竴浜涘疄鐢ㄧ殑[閽╁瓙](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/hook.html)锛屼絾鏈変簺鍦哄悎鐢ㄦ埛鍙兘闇€瑕佸疄鐜颁竴涓柊鐨勯挬瀛愩€傚湪 v1.1.0rc0 涔嬪悗锛孧MDetection3D 鍦ㄨ缁冩椂鏀寔鍩轰簬 MMEngine 鑷畾涔夐挬瀛愩€傚洜姝ょ敤鎴峰彲浠ョ洿鎺ュ湪 mmdet3d 鎴栬€呭熀浜?mmdet3d 鐨勪唬鐮佸簱涓疄鐜伴挬瀛愬苟閫氳繃鏇存敼璁粌閰嶇疆鏉ヤ娇鐢ㄩ挬瀛愩€傝繖閲屾垜浠粰鍑轰竴涓湪 mmdet3d 涓垱寤哄苟浣跨敤鏂伴挬瀛愮殑渚嬪瓙銆?
```python
from mmengine.hooks import Hook

from mmdet3d.registry import HOOKS


@HOOKS.register_module()
class MyHook(Hook):

    def __init__(self, a, b):

    def before_run(self, runner) -> None:

    def after_run(self, runner) -> None:

    def before_train(self, runner) -> None:

    def after_train(self, runner) -> None:

    def before_train_epoch(self, runner) -> None:

    def after_train_epoch(self, runner) -> None:

    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: DATA_BATCH = None) -> None:

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
```

鐢ㄦ埛闇€瑕佹牴鎹挬瀛愮殑鍔熻兘鎸囧畾閽╁瓙鍦ㄦ瘡涓缁冮樁娈垫椂鐨勮涓猴紝鍏蜂綋鍖呮嫭濡備笅闃舵锛歚before_run`锛宍after_run`锛宍before_train`锛宍after_train`锛宍before_train_epoch`锛宍after_train_epoch`锛宍before_train_iter`锛屽拰 `after_train_iter`銆傛湁鏇村鐨勪綅鐐瑰彲浠ユ彃鍏ラ挬瀛愶紝璇︽儏鍙弬鑰?[base hook class](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/hook.py#L9)銆?
#### 2. 娉ㄥ唽鏂伴挬瀛?
鎺ヤ笅鏉ユ垜浠渶瑕佸鍏?`MyHook`銆傚亣璁炬柊閽╁瓙浣嶄簬鏂囦欢 `mmdet3d/engine/hooks/my_hook.py` 涓紝鏈変袱绉嶅疄鐜版柟娉曪細

- 淇敼 `mmdet3d/engine/hooks/__init__.py` 瀵煎叆璇ユā鍧椼€?
  鏂板畾涔夌殑妯″潡搴旇鍦?`mmdet3d/engine/hooks/__init__.py` 涓瀵煎叆锛屼粠鑰岃鎵惧埌骞朵笖琚坊鍔犲埌娉ㄥ唽鍣ㄤ腑锛?
  ```python
  from .my_hook import MyHook
  ```

- 鍦ㄩ厤缃腑浣跨敤 `custom_imports` 鏉ヤ汉涓哄湴瀵煎叆鏂伴挬瀛愩€?
  ```python
  custom_imports = dict(imports=['mmdet3d.engine.hooks.my_hook'], allow_failed_imports=False)
  ```

#### 3. 鏇存敼閰嶇疆鏂囦欢

```python
custom_hooks = [
    dict(type='MyHook', a=a_value, b=b_value)
]
```

鎮ㄥ彲浠ュ皢瀛楁 `priority` 璁剧疆涓?`'NORMAL'` 鎴栬€?`'HIGHEST'` 鏉ヨ缃挬瀛愮殑浼樺厛绾э紝濡備笅鎵€绀猴細

```python
custom_hooks = [
    dict(type='MyHook', a=a_value, b=b_value, priority='NORMAL')
]
```

榛樿鎯呭喌涓嬶紝娉ㄥ唽闃舵閽╁瓙鐨勪紭鍏堢骇涓?`'NORMAL'`銆?
### 浣跨敤 MMDetection3D 涓疄鐜扮殑閽╁瓙

濡傛灉 MMDetection3D 涓凡缁忓疄鐜颁簡璇ラ挬瀛愶紝鎮ㄥ彲浠ョ洿鎺ラ€氳繃鏇存敼閰嶇疆鏂囦欢鏉ヤ娇鐢ㄨ閽╁瓙銆?
#### 渚嬪瓙锛歚DisableObjectSampleHook`

鎴戜滑瀹炵幇浜嗕竴涓悕涓?[DisableObjectSampleHook](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/mmdet3d/engine/hooks/disable_object_sample_hook.py) 鐨勮嚜瀹氫箟閽╁瓙鍦ㄨ缁冮樁娈佃揪鍒版寚瀹?epoch 鍚庣鐢?`ObjectSample` 澧炲己绛栫暐銆?
濡傛灉鏈夐渶瑕佺殑璇濇垜浠彲浠ュ湪閰嶇疆鏂囦欢涓缃畠锛?
```python
custom_hooks = [dict(type='DisableObjectSampleHook', disable_after_epoch=15)]
```

### 鏇存敼榛樿鐨勮繍琛屾椂閽╁瓙

鏈変竴浜涘父鐢ㄧ殑閽╁瓙閫氳繃 `default_hooks` 娉ㄥ唽锛屽畠浠槸锛?
- `IterTimerHook`锛氳閽╁瓙鐢ㄦ潵璁板綍鍔犺浇鏁版嵁鐨勬椂闂?'data_time' 鍜屾ā鍨嬭缁冧竴姝ョ殑鏃堕棿 'time'銆?- `LoggerHook`锛氳閽╁瓙鐢ㄦ潵浠巂鎵ц鍣紙Runner锛塦鐨勪笉鍚岀粍浠舵敹闆嗘棩蹇楀苟灏嗗叾鍐欏叆缁堢锛宩son 鏂囦欢锛宼ensorboard 鍜?wandb 绛夈€?- `ParamSchedulerHook`锛氳閽╁瓙鐢ㄦ潵鏇存柊浼樺寲鍣ㄤ腑鐨勪竴浜涜秴鍙傛暟锛屼緥濡傚涔犵巼鍜屽姩閲忋€?- `CheckpointHook`锛氳閽╁瓙鐢ㄦ潵瀹氭湡鍦颁繚瀛樻鏌ョ偣銆?- `DistSamplerSeedHook`锛氳閽╁瓙鐢ㄦ潵璁剧疆閲囨牱鍜屾壒閲囨牱鐨勭瀛愩€?- `Det3DVisualizationHook`锛氳閽╁瓙鐢ㄦ潵鍙鍖栭獙璇佸拰娴嬭瘯杩囩▼鐨勯娴嬬粨鏋溿€?
`IterTimerHook`锛宍ParamSchedulerHook` 鍜?`DistSamplerSeedHook` 閮藉緢绠€鍗曪紝閫氬父涓嶉渶瑕佷慨鏀癸紝鍥犳姝ゅ鎴戜滑灏嗕粙缁嶅浣曚娇鐢?`LoggerHook`锛宍CheckpointHook` 鍜?`Det3DVisualizationHook`銆?
#### CheckpointHook

闄や簡瀹氭湡鍦颁繚瀛樻鏌ョ偣锛孾`CheckpointHook`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/checkpoint_hook.py#L18) 鎻愪緵浜嗗叾瀹冪殑鍙€夐」渚嬪 `max_keep_ckpts`锛宍save_optimizer` 绛夈€傜敤鎴峰彲浠ヨ缃?`max_keep_ckpts` 鍙繚瀛樺皯閲忕殑妫€鏌ョ偣鎴栬€呴€氳繃 `save_optimizer` 鍐冲畾鏄惁淇濆瓨浼樺寲鍣ㄧ殑鐘舵€併€傚弬鏁扮殑鏇村缁嗚妭璇峰弬鑰僛姝ゅ](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/checkpoint_hook.py#L18)銆?
```python
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_optimizer=True))
```

#### LoggerHook

`LoggerHook` 鍏佽璁剧疆鏃ュ織璁板綍闂撮殧銆傝缁嗕粙缁嶅彲鍙傝€僛鏂囨。](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/logger_hook.py#L19)銆?
```python
default_hooks = dict(logger=dict(type='LoggerHook', interval=50))
```

#### Det3DVisualizationHook

`Det3DVisualizationHook` 浣跨敤 `DetLocalVisualizer` 鏉ュ彲瑙嗗寲棰勬祴缁撴灉锛宍Det3DLocalVisualizer` 鏀寔涓嶅悓鐨勫悗绔紝渚嬪 `TensorboardVisBackend` 鍜?`WandbVisBackend`锛堟洿澶氱粏鑺傝鍙傝€僛鏂囨。](https://github.com/open-mmlab/mmengine/blob/main/mmengine/visualization/vis_backend.py)锛夈€傜敤鎴峰彲浠ユ坊鍔犲涓悗绔潵杩涜鍙鍖栵紝濡備笅鎵€绀恒€?
```python
default_hooks = dict(
    visualization=dict(type='Det3DVisualizationHook', draw=True))

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
```

