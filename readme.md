In this work, we evaluated the authenticity of drone in real time scenario using computer vision based deep tracker.

More information about this work is in this [file](Drone_verification_using_SiamRPN_Tracker.pdf)

#### Python version 3.7 has been used.

#### Following installations are required for working code 

##### Install the below libraries inside your conda environment

```shell
conda install pytorch torchvision -c pytorch

pip install opencv-python imutils pyyaml yacs tqdm colorama matplotlib cython tensorboardX
```

#### For using the code on other platforms except OSX, comment the following line from `drone_tracking.py` 
```
os.environ['KMP_DUPLICATE_LIB_OK']='True'
```

### Run the following code for running the program. 

```python
python drone_tracking.py --config experiments/siamrpn_alex_dwxcorr_otb/config.yaml --snapshot experiments/siamrpn_alex_dwxcorr_otb/model.pth
```

--config flag is for network configuration path

--snapshot flag is for network learned weights

### Reference
Also, we utilized the code from [pysot](https://github.com/STVIR/pysot) for SiamRPN. 
For, more information on SiamRPN, kindly refer to the below work.   
```
@INPROCEEDINGS{8579033,
  author={Li, Bo and Yan, Junjie and Wu, Wei and Zhu, Zheng and Hu, Xiaolin},
  booktitle={2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition}, 
  title={High Performance Visual Tracking with Siamese Region Proposal Network}, 
  year={2018},
  pages={8971-8980},
  doi={10.1109/CVPR.2018.00935}}
```


