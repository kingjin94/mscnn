rm .itter_para.tmp

for dir in ./mscnn-7* ; do 
 for file in $dir/*_train_2*.caffemodel; do

   #echo $dir | grep -o 'mscnn-[^/]*';
   #echo $file | grep -o 'mscnn_[^ ]*.caffemodel';

   echo ${dir#./} ${file##[^ ]*/} >> .itter_para.tmp
  

 done;
done

cat .itter_para.tmp | xargs -n 2 -P 8 python run_elementary_detection.py | tee -a eval_log.txt;
rm .itter_para.tmp
