### Self-produced Guidance for Weakly-supervised Object Localization


![](figs/fig1-1.png)

# Train
We finetune the SPG model on the ILSVRC dataset.  
```
cd scripts
sh train_imagenet_full_v5.sh
```


# Test
 Download the pretrined model at GoogleDrive(https://drive.google.com/open?id=1EwRuqfGASarGidutnYB8rXLSuzYpEoSM).

 Use the test script to generate attention maps.
```
cd scripts
sh val_imagenet_full.sh
```

![](figs/imagenet-box-1.png)

### Citation
If you find this code helpful, please consider to cite this paper:
```
@inproceedings{zhang2018self,
  title={Self-produced Guidance for Weakly-supervised Object Localization},
  author={Zhang, Xiaolin and Kang, Guoliang and Wei, Yunchao and Yang, Yi and Huang, Thomas},
  booktitle={European Conference on Computer Vision},
  year={2018},
  organization={Springer}
}
```
