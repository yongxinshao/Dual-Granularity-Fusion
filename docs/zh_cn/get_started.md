# 寮€濮嬩綘鐨勭涓€姝?
## 渚濊禆

鍦ㄦ湰鑺備腑锛屾垜浠皢灞曠ず濡備綍浣跨敤 PyTorch 鍑嗗鐜銆?
MMDetection3D 鏀寔鍦?Linux锛學indows锛堝疄楠屾€ф敮鎸侊級锛孧acOS 涓婅繍琛岋紝瀹冮渶瑕?Python 3.7 浠ヤ笂锛孋UDA 9.2 浠ヤ笂鍜?PyTorch 1.6 浠ヤ笂銆?
```{note}
濡傛灉鎮ㄥ PyTorch 鏈夌粡楠屽苟涓斿凡缁忓畨瑁呬簡瀹冿紝鎮ㄥ彲浠ョ洿鎺ヨ烦杞埌[涓嬩竴灏忚妭](#瀹夎娴佺▼)銆傚惁鍒欙紝鎮ㄥ彲浠ユ寜鐓т笅杩版楠よ繘琛屽噯澶囥€?```

**姝ラ 0.** 浠嶽瀹樻柟缃戠珯](https://docs.conda.io/en/latest/miniconda.html)涓嬭浇骞跺畨瑁?Miniconda銆?
**姝ラ 1.** 鍒涘缓骞舵縺娲讳竴涓?conda 鐜銆?
```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**姝ラ 2.** 鍩轰簬 [PyTorch 瀹樻柟璇存槑](https://pytorch.org/get-started/locally/)瀹夎 PyTorch锛屼緥濡傦細

鍦?GPU 骞冲彴涓婏細

```shell
conda install pytorch torchvision -c pytorch
```

鍦?CPU 骞冲彴涓婏細

```shell
conda install pytorch torchvision cpuonly -c pytorch
```

## 瀹夎娴佺▼

鎴戜滑鎺ㄨ崘鐢ㄦ埛鍙傜収鎴戜滑鐨勬渶浣冲疄璺靛畨瑁?MMDetection3D銆備笉杩囷紝鏁翠釜杩囩▼涔熸槸鍙畾鍒跺寲鐨勶紝鏇村淇℃伅璇峰弬鑰僛鑷畾涔夊畨瑁匽(#鑷畾涔夊畨瑁?绔犺妭銆?
### 鏈€浣冲疄璺?
**姝ラ 0.** 浣跨敤 [MIM](https://github.com/open-mmlab/mim) 瀹夎 [MMEngine](https://github.com/open-mmlab/mmengine)锛孾MMCV](https://github.com/open-mmlab/mmcv) 鍜?[MMDetection](https://github.com/open-mmlab/mmdetection)銆?
```shell
pip install -U openmim
mim install mmengine
mim install 'mmcv>=2.0.0rc4'
mim install 'mmdet>=3.0.0'
```

**娉ㄦ剰**锛氬湪 MMCV-v2.x 涓紝`mmcv-full` 鏀瑰悕涓?`mmcv`锛屽鏋滄偍鎯冲畨瑁呬笉鍖呭惈 CUDA 绠楀瓙鐨?`mmcv`锛屾偍鍙互浣跨敤 `mim install "mmcv-lite>=2.0.0rc4"` 瀹夎绮剧畝鐗堛€?
**姝ラ 1.** 瀹夎 MMDetection3D銆?
鏂规 a锛氬鏋滄偍寮€鍙戝苟鐩存帴杩愯 mmdet3d锛屼粠婧愮爜瀹夎瀹冿細

```shell
git clone https://github.com/open-mmlab/mmdetection3d.git -b dev-1.x
# "-b dev-1.x" 琛ㄧず鍒囨崲鍒?`dev-1.x` 鍒嗘敮銆?cd mmdetection3d
pip install -v -e .
# "-v" 鎸囪缁嗚鏄庯紝鎴栨洿澶氱殑杈撳嚭
# "-e" 琛ㄧず鍦ㄥ彲缂栬緫妯″紡涓嬪畨瑁呴」鐩紝鍥犳瀵逛唬鐮佹墍鍋氱殑浠讳綍鏈湴淇敼閮戒細鐢熸晥锛屼粠鑰屾棤闇€閲嶆柊瀹夎銆?```

鏂规 b锛氬鏋滄偍灏?mmdet3d 浣滀负渚濊禆鎴栫涓夋柟 Python 鍖呬娇鐢紝浣跨敤 MIM 瀹夎锛?
```shell
mim install "mmdet3d>=1.1.0rc0"
```

娉ㄦ剰锛?
1. 濡傛灉鎮ㄥ笇鏈涗娇鐢?`opencv-python-headless` 鑰屼笉鏄?`opencv-python`锛屾偍鍙互鍦ㄥ畨瑁?MMCV 涔嬪墠瀹夎瀹冦€?
2. 涓€浜涘畨瑁呬緷璧栨槸鍙€夌殑銆傜畝鍗曞湴杩愯 `pip install -v -e .` 灏嗕細瀹夎鏈€浣庤繍琛岃姹傜殑鐗堟湰銆傚鏋滄兂瑕佷娇鐢ㄤ竴浜涘彲閫変緷璧栭」锛屼緥濡?`albumentations` 鍜?`imagecorruptions`锛屽彲浠ヤ娇鐢?`pip install -r requirements/optional.txt` 杩涜鎵嬪姩瀹夎锛屾垨鑰呭湪浣跨敤 `pip` 鏃舵寚瀹氭墍闇€鐨勯檮鍔犲姛鑳斤紙渚嬪 `pip install -v -e .[optional]`锛夛紝鏀寔闄勫姞鍔熻兘鐨勬湁鏁堥敭鍊煎寘鎷?`all`銆乣tests`銆乣build` 浠ュ強 `optional`銆?
   鎴戜滑宸茬粡鏀寔 `spconv 2.0`銆傚鏋滅敤鎴峰凡缁忓畨瑁?`spconv 2.0`锛屼唬鐮佷細榛樿浣跨敤 `spconv 2.0`锛屽畠浼氭瘮鍘熺敓 `mmcv spconv` 浣跨敤鏇村皯鐨?GPU 鍐呭瓨銆傜敤鎴峰彲浠ヤ娇鐢ㄤ笅鍒楃殑鍛戒护鏉ュ畨瑁?`spconv 2.0`锛?
   ```shell
   pip install cumm-cuxxx
   pip install spconv-cuxxx
   ```

   `xxx` 琛ㄧず鐜涓殑 CUDA 鐗堟湰銆?
   渚嬪锛屼娇鐢?CUDA 10.2锛屽搴斿懡浠ゆ槸 `pip install cumm-cu102 && pip install spconv-cu102`銆?
   鏀寔鐨?CUDA 鐗堟湰鍖呮嫭 10.2锛?1.1锛?1.3 鍜?11.4銆傜敤鎴蜂篃鍙互閫氳繃婧愮爜缂栬瘧鏉ュ畨瑁呫€傛洿澶氱粏鑺傝鍙傝€僛spconv v2.x](https://github.com/traveller59/spconv)銆?
   鎴戜滑涔熸敮鎸?`Minkowski Engine` 浣滀负绋€鐤忓嵎绉殑鍚庣銆傚鏋滈渶瑕侊紝璇峰弬鑰僛瀹夎鎸囧崡](https://github.com/NVIDIA/MinkowskiEngine#installation) 鎴栬€呬娇鐢?`pip` 鏉ュ畨瑁咃細

   ```shell
   conda install openblas-devel -c anaconda
   export CPLUS_INCLUDE_PATH=CPLUS_INCLUDE_PATH:${YOUR_CONDA_ENVS_DIR}/include
   # replace ${YOUR_CONDA_ENVS_DIR} to your anaconda environment path e.g. `/home/username/anaconda3/envs/openmmlab`.
   pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=/opt/conda/include" --install-option="--blas=openblas"
   ```

   鎴戜滑杩樻敮鎸?`Torchsparse` 浣滀负绋€鐤忓嵎绉殑鍚庣銆傚鏋滈渶瑕侊紝璇峰弬鑰僛瀹夎鎸囧崡](https://github.com/mit-han-lab/torchsparse#installation) 鎴栬€呬娇鐢?`pip` 鏉ュ畨瑁咃細

   ```shell
   sudo apt install libsparsehash-dev
   pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
   ```

   鎴栬€呴€氳繃浠ヤ笅瀹夎缁曡繃sudo鏉冮檺

   ```shell
   conda install -c bioconda sparsehash
   export CPLUS_INCLUDE_PATH=CPLUS_INCLUDE_PATH:${YOUR_CONDA_ENVS_DIR}/include
    # replace ${YOUR_CONDA_ENVS_DIR} to your anaconda environment path e.g. `/home/username/anaconda3/envs/openmmlab`.
   pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
   ```

3. 鎴戜滑鐨勪唬鐮佺洰鍓嶄笉鑳藉湪鍙湁 CPU 鐨勭幆澧冿紙CUDA 涓嶅彲鐢級涓嬬紪璇戙€?
### 楠岃瘉瀹夎

涓轰簡楠岃瘉 MMDetection3D 鏄惁瀹夎姝ｇ‘锛屾垜浠彁渚涗簡涓€浜涚ず渚嬩唬鐮佹潵鎵ц妯″瀷鎺ㄧ悊銆?
**姝ラ 1.** 鎴戜滑闇€瑕佷笅杞介厤缃枃浠跺拰妯″瀷鏉冮噸鏂囦欢銆?
```shell
mim download mmdet3d --config pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car --dest .
```

涓嬭浇灏嗛渶瑕佸嚑绉掗挓鎴栨洿闀挎椂闂达紝杩欏彇鍐充簬鎮ㄧ殑缃戠粶鐜銆傚畬鎴愬悗锛屾偍浼氬湪褰撳墠鏂囦欢澶逛腑鍙戠幇涓や釜鏂囦欢 `pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py` 鍜?`hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth`銆?
**姝ラ 2.** 鎺ㄧ悊楠岃瘉銆?
鏂规 a锛氬鏋滄偍浠庢簮鐮佸畨瑁?MMDetection3D锛岄偅涔堢洿鎺ヨ繍琛屼互涓嬪懡浠よ繘琛岄獙璇侊細

```shell
python demo/pcd_demo.py demo/data/kitti/000008.bin pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth --show
```

鎮ㄤ細鐪嬪埌涓€涓甫鏈夌偣浜戠殑鍙鍖栫晫闈紝鍏朵腑鍖呭惈鏈夊湪姹借溅涓婄粯鍒剁殑妫€娴嬫銆?
**娉ㄦ剰**锛?
濡傛灉浣犲湪娌℃湁鏄剧ず璁惧鐨勬湇鍔″櫒涓婂畨瑁?MMDetection3D 锛屼綘鍙互蹇界暐 `--show` 鍙傛暟銆侱emo 浠嶄細灏嗛娴嬬粨鏋滀繚瀛樺埌 `outputs/pred/000008.json` 鏂囦欢涓€?
**娉ㄦ剰**锛?
濡傛灉鎮ㄦ兂杈撳叆涓€涓?`.ply` 鏂囦欢锛屾偍鍙互浣跨敤濡備笅鍑芥暟灏嗗畠杞崲鎴?`.bin` 鏍煎紡銆傜劧鍚庢偍鍙互浣跨敤杞寲鐨?`.bin` 鏂囦欢鏉ヨ繍琛屾牱渚嬨€傝娉ㄦ剰鍦ㄤ娇鐢ㄦ鑴氭湰涔嬪墠锛屾偍闇€瑕佸畨瑁?`pandas` 鍜?`plyfile`銆傝繖涓嚱鏁颁篃鍙互鐢ㄤ簬璁粌 `ply 鏁版嵁`鏃朵綔涓烘暟鎹澶勭悊鏉ヤ娇鐢ㄣ€?
```python
import numpy as np
import pandas as pd
from plyfile import PlyData

def convert_ply(input_path, output_path):
    plydata = PlyData.read(input_path)  # 璇诲彇鏂囦欢
    data = plydata.elements[0].data  # 璇诲彇鏁版嵁
    data_pd = pd.DataFrame(data)  # 杞崲鎴?DataFrame
    data_np = np.zeros(data_pd.shape, dtype=np.float)  # 鍒濆鍖栨暟缁勬潵瀛樺偍鏁版嵁
    property_names = data[0].dtype.names  # 璇诲彇灞炴€у悕绉?    for i, name in enumerate(
            property_names):  # 閫氳繃灞炴€ц鍙栨暟鎹?        data_np[:, i] = data_pd[name]
    data_np.astype(np.float32).tofile(output_path)
```

渚嬪锛?
```python
convert_ply('./test.ply', './test.bin')
```

濡傛灉鎮ㄦ湁鍏朵粬鏍煎紡鐨勭偣浜戞暟鎹紙`.off`锛宍.obj` 绛夛級锛屾偍鍙互浣跨敤 `trimesh` 灏嗗畠浠浆鍖栨垚 `.ply`銆?
```python
import trimesh

def to_ply(input_path, output_path, original_type):
    mesh = trimesh.load(input_path, file_type=original_type)  # 璇诲彇鏂囦欢
    mesh.export(output_path, file_type='ply')  # 杞崲鎴?ply
```

渚嬪锛?
```python
to_ply('./test.obj', './test.ply', 'obj')
```

鏂规 b锛氬鏋滄偍浣跨敤 MIM 瀹夎 MMDetection3D锛岄偅涔堝彲浠ユ墦寮€鎮ㄧ殑 Python 瑙ｆ瀽鍣紝澶嶅埗骞剁矘璐翠互涓嬩唬鐮侊細

```python
from mmdet3d.apis import init_model, inference_detector

config_file = 'pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py'
checkpoint_file = 'hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth'
model = init_model(config_file, checkpoint_file)
inference_detector(model, 'demo/data/kitti/000008.bin')
```

鎮ㄥ皢浼氱湅鍒颁竴涓寘鍚?`Det3DDataSample` 鐨勫垪琛紝棰勬祴缁撴灉鍦?`pred_instances_3d` 閲岄潰锛屽寘鍚湁妫€娴嬫锛岀被鍒拰寰楀垎銆?
### 鑷畾涔夊畨瑁?
#### CUDA 鐗堟湰

鍦ㄥ畨瑁?PyTorch 鏃讹紝鎮ㄩ渶瑕佹寚瀹?CUDA 鐨勭増鏈€傚鏋滄偍涓嶆竻妤氬簲璇ラ€夋嫨鍝竴涓紝璇烽伒寰垜浠殑寤鸿锛?
- 瀵逛簬 Ampere 鏋舵瀯鐨?NVIDIA GPU锛屼緥濡?GeForce 30 绯诲垪浠ュ強 NVIDIA A100锛孋UDA 11 鏄繀闇€鐨勩€?- 瀵逛簬鏇存棭鐨?NVIDIA GPU锛孋UDA 11 鏄悜鍚庡吋瀹圭殑锛屼絾 CUDA 10.2 鎻愪緵鏇村ソ鐨勫吋瀹规€э紝骞朵笖鏇磋交閲忋€?
璇风‘淇?GPU 椹卞姩鐗堟湰婊¤冻鏈€浣庣殑鐗堟湰闇€姹傘€傛洿澶氫俊鎭鍙傝€冩[琛ㄦ牸](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions)銆?
```{note}
濡傛灉鎮ㄩ伒寰垜浠殑鏈€浣冲疄璺碉紝鎮ㄥ彧闇€瑕佸畨瑁?CUDA 杩愯搴擄紝杩欐槸鍥犱负涓嶉渶瑕佸湪鏈湴缂栬瘧 CUDA 浠ｇ爜銆備絾濡傛灉鎮ㄥ笇鏈涗粠婧愮爜缂栬瘧 MMCV锛屾垨鑰呭紑鍙戝叾浠?CUDA 绠楀瓙锛岄偅涔堟偍闇€瑕佷粠 NVIDIA 鐨刐瀹樼綉](https://developer.nvidia.com/cuda-downloads)瀹夎瀹屾暣鐨?CUDA 宸ュ叿閾撅紝骞朵笖璇ョ増鏈簲璇ヤ笌 PyTorch 鐨?CUDA 鐗堟湰鐩稿尮閰嶏紝姣斿鍦?`conda install` 鎸囦护閲屾寚瀹?cudatoolkit 鐗堟湰銆?```

#### 涓嶉€氳繃 MIM 瀹夎 MMEngine

濡傛灉鎯宠浣跨敤 pip 鑰屼笉鏄?MIM 瀹夎 MMEngine锛岃鍙傝€?[MMEngine 瀹夎鎸囧崡](https://mmengine.readthedocs.io/zh_CN/latest/get_started/installation.html)銆?
渚嬪锛屾偍鍙互閫氳繃浠ヤ笅鎸囦护瀹夎 MMEngine锛?
```shell
pip install mmengine
```

#### 涓嶉€氳繃 MIM 瀹夎 MMCV

MMCV 鍖呭惈 C++ 鍜?CUDA 鎷撳睍锛屽洜姝ゅ叾瀵?PyTorch 鐨勪緷璧栨洿澶嶆潅銆侻IM 浼氳嚜鍔ㄨВ鍐虫绫讳緷璧栧叧绯诲苟浣垮畨瑁呮洿瀹规槗銆備絾杩欎笉鏄繀闇€鐨勩€?
濡傛灉鎯宠浣跨敤 pip 鑰屼笉鏄?MIM 瀹夎 MMCV锛岃鍙傝€?[MMCV 瀹夎鎸囧崡](https://mmcv.readthedocs.io/zh_CN/2.x/get_started/installation.html)銆傝繖闇€瑕佺敤鎸囧畾 url 鐨勫舰寮忔墜鍔ㄦ寚瀹氬搴旂殑 PyTorch 鍜?CUDA 鐗堟湰銆?
渚嬪锛屼笅杩版寚浠ゅ皢浼氬畨瑁呭熀浜?PyTorch 1.12.x 鍜?CUDA 11.6 缂栬瘧鐨?MMCV锛?
```shell
pip install "mmcv>=2.0.0rc4" -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.0/index.html
```

#### 鍦?Google Colab 涓畨瑁?
[Google Colab](https://colab.research.google.com/) 閫氬父宸茬粡瀹夎浜?PyTorch锛屽洜姝ゆ垜浠彧闇€瑕佺敤濡備笅鍛戒护瀹夎 MMEngine锛孧MCV锛孧MDetection 鍜?MMDetection3D 鍗冲彲銆?
**姝ラ 1.** 浣跨敤 [MIM](https://github.com/open-mmlab/mim) 瀹夎 [MMEngine](https://github.com/open-mmlab/mmengine)锛孾MMCV](https://github.com/open-mmlab/mmcv) 鍜?[MMDetection](https://github.com/open-mmlab/mmdetection)銆?
```shell
!pip3 install openmim
!mim install mmengine
!mim install "mmcv>=2.0.0rc4,<2.1.0"
!mim install "mmdet>=3.0.0,<3.1.0"
```

**姝ラ 2.** 浠庢簮鐮佸畨瑁?MMDetection3D銆?
```shell
!git clone https://github.com/open-mmlab/mmdetection3d.git -b dev-1.x
%cd mmdetection3d
!pip install -e .
```

**姝ラ 3.** 楠岃瘉瀹夎鏄惁鎴愬姛銆?
```python
import mmdet3d
print(mmdet3d.__version__)
# 棰勬湡杈撳嚭锛?.1.0rc0 鎴栧叾瀹冪増鏈彿銆?```

```{note}
鍦?Jupyter Notebook 涓紝鎰熷徆鍙?`!` 鐢ㄤ簬鎵ц澶栭儴鍛戒护锛岃€?`%cd` 鏄竴涓猍榄旀湳鍛戒护](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-cd)锛岀敤浜庡垏鎹?Python 鐨勫伐浣滆矾寰勩€?```

#### 閫氳繃 Docker 浣跨敤 MMDetection3D

鎴戜滑鎻愪緵浜?[Dockerfile](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/docker/Dockerfile) 鏉ユ瀯寤轰竴涓暅鍍忋€傝纭繚鎮ㄧ殑 [docker 鐗堟湰](https://docs.docker.com/engine/install/) >= 19.03銆?
```shell
# 鍩轰簬 PyTorch 1.9锛孋UDA 11.1 鏋勫缓闀滃儚
# 濡傛灉鎮ㄦ兂瑕佸叾浠栫増鏈紝鍙渶瑕佷慨鏀?Dockerfile
docker build -t mmdetection3d docker/
```

鐢ㄤ互涓嬪懡浠よ繍琛?Docker 闀滃儚锛?
```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmdetection3d/data mmdetection3d
```

### 鏁呴殰鎺掗櫎

濡傛灉鎮ㄥ湪瀹夎杩囩▼涓亣鍒颁竴浜涢棶棰橈紝璇峰厛鍙傝€?[FAQ](notes/faq.md) 椤甸潰銆傚鏋滄病鏈夋壘鍒板搴旂殑瑙ｅ喅鏂规锛屾偍涔熷彲浠ュ湪 GitHub [鎻愪竴涓棶棰榏(https://github.com/open-mmlab/mmdetection3d/issues/new/choose)銆?
### 浣跨敤澶氫釜 MMDetection3D 鐗堟湰杩涜寮€鍙?
璁粌鍜屾祴璇曠殑鑴氭湰宸茬粡鍦?`PYTHONPATH` 涓繘琛屼簡淇敼锛屼互纭繚鑴氭湰浣跨敤褰撳墠鐩綍涓殑 MMDetection3D銆?
瑕佷娇鐜涓畨瑁呴粯璁ょ増鏈殑 MMDetection3D 鑰屼笉鏄綋鍓嶆鍦ㄤ娇鐢ㄧ殑锛屽彲浠ュ垹闄ゅ嚭鐜板湪鐩稿叧鑴氭湰涓殑浠ｇ爜锛?
```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```

