
#++++++++++ Protocol-A ++++++++++++++
### Inception, Protocol-A, L2-norm
# bypass_prn=True
python main.py --testcase_ID=A_L2_XNorm_0 \
               --bypass_prn=True --perturb_en=True --perturb_mode=0 \
               --data_dir='/media/jianl/TOSHIBA-EXT/Datasets/ImageNet/ILSVRC2012_img_val_Rearrange_ORIGINAL/' \
               --pert_test_dir='inception_L2_Pert_TEST/' \
               #--pretrained_model='../models/generative_2017-11-09_14-43-08/model-inception.ckpt-1600'
               --pretrained_model='../models/generative_2017-11-11_00-29-01/model-inception.ckpt-2200'

#bypass_prn=False, perturb_mode=0
python main.py --testcase_ID=A_L2_XNorm_1 \
               --bypass_prn=False --perturb_en=True --perturb_mode=0 \
               --data_dir='/media/jianl/TOSHIBA-EXT/Datasets/ImageNet/ILSVRC2012_img_val_Rearrange_ORIGINAL/' \
               --pert_test_dir='inception_L2_Pert_TEST/' \
               --pretrained_model='../models/generative_2017-11-11_00-29-01/model-inception.ckpt-2200'

#bypass_prn=False, perturb_mode=1
python main.py --testcase_ID=A_L2_XNorm_2 \
               --bypass_prn=False --perturb_en=True --perturb_mode=1 \
               --data_dir='/media/jianl/TOSHIBA-EXT/Datasets/ImageNet/ILSVRC2012_img_val_Rearrange_ORIGINAL/' \
               --pert_test_dir='inception_L2_Pert_TEST/' \
               --pretrained_model='../models/generative_2017-11-11_00-29-01/model-inception.ckpt-2200'

#bypass_prn=False, perturb_en=False
python main.py --testcase_ID=A_L2_XNorm_3 \
               --bypass_prn=False --perturb_en=False --perturb_mode=0 \
               --data_dir='/media/jianl/TOSHIBA-EXT/Datasets/ImageNet/ILSVRC2012_img_val_Rearrange_ORIGINAL/' \
               --pert_test_dir='inception_L2_Pert_TEST/' \
               --pretrained_model='../models/generative_2017-11-11_00-29-01/model-inception.ckpt-2200'


### Inception, Protocol-A, Linf-norm
# bypass_prn=True
python main.py --testcase_ID=A_Linf_XNorm_0 \
               --bypass_prn=True --perturb_en=True --perturb_mode=0 \
               --data_dir='/media/jianl/TOSHIBA-EXT/Datasets/ImageNet/ILSVRC2012_img_val_Rearrange_ORIGINAL/' \
               --pert_test_dir='inception_Linf_Pert_TEST/' \
               #--pretrained_model='../models/generative_2017-11-11_00-29-01/model-inception.ckpt-2200'
               --pretrained_model='../models/generative_2017-11-09_14-43-08/model-inception.ckpt-1600'

#bypass_prn=False, perturb_mode=0
python main.py --testcase_ID=A_Linf_XNorm_1 \
               --bypass_prn=False --perturb_en=True --perturb_mode=0 \
               --data_dir='/media/jianl/TOSHIBA-EXT/Datasets/ImageNet/ILSVRC2012_img_val_Rearrange_ORIGINAL/' \
               --pert_test_dir='inception_Linf_Pert_TEST/' \
               --pretrained_model='../models/generative_2017-11-09_14-43-08/model-inception.ckpt-1600'

#bypass_prn=False, perturb_mode=1
python main.py --testcase_ID=A_Linf_XNorm_2 \
               --bypass_prn=False --perturb_en=True --perturb_mode=1 \
               --data_dir='/media/jianl/TOSHIBA-EXT/Datasets/ImageNet/ILSVRC2012_img_val_Rearrange_ORIGINAL/' \
               --pert_test_dir='inception_Linf_Pert_TEST/' \
               --pretrained_model='../models/generative_2017-11-09_14-43-08/model-inception.ckpt-1600'

#bypass_prn=False, perturb_en=False
python main.py --testcase_ID=A_Linf_XNorm_3 \
               --bypass_prn=False --perturb_en=False --perturb_mode=0 \
               --data_dir='/media/jianl/TOSHIBA-EXT/Datasets/ImageNet/ILSVRC2012_img_val_Rearrange_ORIGINAL/' \
               --pert_test_dir='inception_Linf_Pert_TEST/' \
               --pretrained_model='../models/generative_2017-11-09_14-43-08/model-inception.ckpt-1600'

#++++++++++ Protocol-B ++++++++++++++
### Inception, Protocol-B, L2-norm
# bypass_prn=True
python main.py --testcase_ID=B_L2_XNorm_0 \
               --bypass_prn=True --perturb_en=True --perturb_mode=0 --eval_protocol=B \
               --data_dir='/media/jianl/TOSHIBA-EXT/Datasets/ImageNet/ILSVRC2012_img_val_Rearrange_ORIGINAL_Inception_Correct_Prediction/' \
               --pert_test_dir='inception_L2_Pert_TEST/' \
               #--pretrained_model='../models/generative_2017-11-09_14-43-08/model-inception.ckpt-1600'
               --pretrained_model='../models/generative_2017-11-11_00-29-01/model-inception.ckpt-2200'

#bypass_prn=False, perturb_mode=0
python main.py --testcase_ID=B_L2_XNorm_1 \
               --bypass_prn=False --perturb_en=True --perturb_mode=0 --eval_protocol=B \
               --data_dir='/media/jianl/TOSHIBA-EXT/Datasets/ImageNet/ILSVRC2012_img_val_Rearrange_ORIGINAL_Inception_Correct_Prediction/' \
               --pert_test_dir='inception_L2_Pert_TEST/' \
               --pretrained_model='../models/generative_2017-11-11_00-29-01/model-inception.ckpt-2200'

#bypass_prn=False, perturb_mode=1
python main.py --testcase_ID=B_L2_XNorm_2 \
               --bypass_prn=False --perturb_en=True --perturb_mode=1 --eval_protocol=B \
               --data_dir='/media/jianl/TOSHIBA-EXT/Datasets/ImageNet/ILSVRC2012_img_val_Rearrange_ORIGINAL_Inception_Correct_Prediction/' \
               --pert_test_dir='inception_L2_Pert_TEST/' \
               --pretrained_model='../models/generative_2017-11-11_00-29-01/model-inception.ckpt-2200'

#bypass_prn=False, perturb_en=False
python main.py --testcase_ID=B_L2_XNorm_3 \
               --bypass_prn=False --perturb_en=False --perturb_mode=0 --eval_protocol=B \
               --data_dir='/media/jianl/TOSHIBA-EXT/Datasets/ImageNet/ILSVRC2012_img_val_Rearrange_ORIGINAL_Inception_Correct_Prediction/' \
               --pert_test_dir='inception_L2_Pert_TEST/' \
               --pretrained_model='../models/generative_2017-11-11_00-29-01/model-inception.ckpt-2200'

### Inception, Protocol-B, Linf-norm
# bypass_prn=True
python main.py --testcase_ID=B_Linf_XNorm_0 \
               --bypass_prn=True --perturb_en=True --perturb_mode=0 --eval_protocol=B \
               --data_dir='/media/jianl/TOSHIBA-EXT/Datasets/ImageNet/ILSVRC2012_img_val_Rearrange_ORIGINAL_Inception_Correct_Prediction/' \
               --pert_test_dir='inception_Linf_Pert_TEST/' \
               #--pretrained_model='../models/generative_2017-11-11_00-29-01/model-inception.ckpt-2200'
               --pretrained_model='../models/generative_2017-11-09_14-43-08/model-inception.ckpt-1600'

#bypass_prn=False, perturb_mode=0
python main.py --testcase_ID=B_Linf_XNorm_1 \
               --bypass_prn=False --perturb_en=True --perturb_mode=0 --eval_protocol=B \
               --data_dir='/media/jianl/TOSHIBA-EXT/Datasets/ImageNet/ILSVRC2012_img_val_Rearrange_ORIGINAL_Inception_Correct_Prediction/' \
               --pert_test_dir='inception_Linf_Pert_TEST/' \
               --pretrained_model='../models/generative_2017-11-09_14-43-08/model-inception.ckpt-1600'

#bypass_prn=False, perturb_mode=1
python main.py --testcase_ID=B_Linf_XNorm_2 \
               --bypass_prn=False --perturb_en=True --perturb_mode=1 --eval_protocol=B \
               --data_dir='/media/jianl/TOSHIBA-EXT/Datasets/ImageNet/ILSVRC2012_img_val_Rearrange_ORIGINAL_Inception_Correct_Prediction/' \
               --pert_test_dir='inception_Linf_Pert_TEST/' \
               --pretrained_model='../models/generative_2017-11-09_14-43-08/model-inception.ckpt-1600'

#bypass_prn=False, perturb_en=False
python main.py --testcase_ID=B_Linf_XNorm_3 \
               --bypass_prn=False --perturb_en=False --perturb_mode=0 --eval_protocol=B \
               --data_dir='/media/jianl/TOSHIBA-EXT/Datasets/ImageNet/ILSVRC2012_img_val_Rearrange_ORIGINAL_Inception_Correct_Prediction/' \
               --pert_test_dir='inception_Linf_Pert_TEST/' \
               --pretrained_model='../models/generative_2017-11-09_14-43-08/model-inception.ckpt-1600'



