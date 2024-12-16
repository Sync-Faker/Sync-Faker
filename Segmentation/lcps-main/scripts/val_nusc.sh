cd ..

nohup python val.py -c configs/pa_po_nuscenes_val.yaml -l nusc_val_falsetime_onlyfilename_2.log --resume > nohup_nusc_val_falsetime_onlyfilename_2.log 2>&1 &