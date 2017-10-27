Manual MSCNN {#app:man}
============

The following sections will describe how one may reproduce the
experiments from my Bachelor Thesis. It will list the required software and data and show how to use them step by step.

Required Data and Software
--------------------------

First of all one will need to install the software that ran the
experiments. Major parts of it are in the following git repository in
the form of Python and MATLAB scripts as well as the C++/ CUDA source
code of Caffe and MS-CNN: <https://github.com/kingjin94/mscnn.git>\
The git repository is expected to be cloned into &lt;ThesisRoot&gt;/.
Now one has two choices, either compile MS-CNN oneself or *preferably*
one can build the Docker-image which runs MS-CNN. If one wants to
compile the source by himself please follow the description given by
MS-CNN in their git repository (<https://github.com/zhaoweicai/mscnn>)
and note that the needed source is already within &lt;ThesisRoot&gt;/.\
If one chooses the Docker-image, first ensure that NVIDIA-Docker is
installed. The process is described here:
<https://github.com/NVIDIA/nvidia-docker>. After installing
NVIDIA-Docker please build the Docker-image with the dockerfile found in
&lt;ThesisRoot&gt;/docker/standalone/gpu/ and label it caffe:MSCNN.\
Regardless of the chosen way to install the software additionally the
used datasets are needed:

-   KITTI Object Detection Evaluation 2012  [http://www.cvlibs.net/datasets/kitti/eval_object.php]: Left color
    images[^1] and Labels[^2]

-   DitM  [https://fcav.engin.umich.edu/sim-dataset/]: 200k Archive images[^3] or Subset used in this
    thesis[^4]

-   DitM labels in KITTI format:
    &lt;ThesisRoot&gt;/data/DitM/label\_2\_DitM.tar.gz

-   Subsets (to be found under:
    &lt;ThesisRoot&gt;/data/{KITTI|DitM}/ImageSets/)

-   Windowfiles (to be found under:
    &lt;ThesisRoot&gt;/data/{KITTI|DitM}/window\_files/)

-   VGG\_16  [http://www.robots.ox.ac.uk/~vgg/research/very_deep/]: <https://goo.gl/LCzWV8>, to be saved in
    &lt;models&gt;/

Both datasets are expected to be in folders named &lt;KITTIRoot&gt;/ and
&lt;DitMRoot&gt;/ respectively. The KITTI dataset is the blueprint for
the folder structure, therefore some alterations to DitM have to be
made:

-   Move the images from the DitM 200k or the smaller subset to the
    folder &lt;DitMRoot&gt;/image\_2/

-   Extract the labels form DitM, found in
    &lt;ThesisRoot&gt;/data/DitM/label\_2\_DitM.tar.gz, to
    &lt;DitMRoot&gt;/label\_2/

-   If one uses the 200k archive: Resize the images by first of all
    rescaling to 1280 width (with constant aspect ratio) and than
    cropping to a height of 384 by equally cutting off top and bottom.
    Furthermore convert them to .png

After the datasets are set up one has to adapt the evaluation scripts to
the setup:

-   In &lt;ThesisRoot&gt;/examples/{KITTI|DitM}/evalFunc.m adapt line 20
    to the appropriate ground-truth directory, which is
    &lt;KITTIRoot&gt;/training/label\_2/ or &lt;DitMRoot&gt;/label\_2/
    respectively.

-   In &lt;ThesisRoot&gt;/examples/KITTI/image\_size.m change the line 2
    to &lt;KITTIRoot&gt;/

-   Compile evaluate\_object.cpp in
    &lt;ThesisRoot&gt;/examples/kitti\_result/eval/ with g++ or another
    C++ compiler; name the program evaluate\_object

Training the CNNs
-----------------

Training with the provided docker image is quiet easy. One has to first
run the images with the data mounted at the right place. This is done
by:

For KITTI:

``` {.bash language="bash"}
sudo nvidia-docker run -ti -v <KITTIRoot>/:/home/data/KITTI/ \
-v <ThesisRoot>/examples/:/home/mscnn/examples/ \
-v <ThesisRoot>/data/:/home/mscnn/data/ \
-v <models>/:/home/mscnn/models/ caffe:MSCNN
```

For DitM:

``` {.bash language="bash"}
sudo nvidia-docker run -ti -v <DitMRoot>/:/home/data/DitM \
-v <ThesisRoot>/examples/:/home/mscnn/examples/ \
-v <ThesisRoot>/data/:/home/mscnn/data/ \
-v <models>/:/home/mscnn/models/ caffe:MSCNN
```

Within the docker container one finds a folder structure similar to
&lt;ThesisRoot&gt;/. To run a training session go to
examples/{kitti\_car|DitM}/&lt;OneExperiment&gt;/. Within there is
mscnn\_train.sh which runs the training. Please adapt the script to the
number of GPUs by changing the –gpu tag to the IDs of the ones to be
used.\
A log of the progress will be written to stdout, which one may want to
save to a log file. Lines with “Loss = ” within give a good sense of
training progress. After the training is done one will find a file named
mscnn\_kitti\_train\_2nd\_iter\_25000.caffemodel within the folder which
are the trained weights.

Evaluation of Results
---------------------

The evaluation of the trained networks with the dockerimages is a two
stage process, allowing the more demanding part (inference) to be run on
a powerful server off-site if necessary. The docker-image handles
forwarding of images through the trained network and produces
intermediate output in form of .tar.gz archives. They may than be
analyzed with the MATLAB script provided in
&lt;ThesisRoot&gt;/examples/{kitti\_car|DitM}/.\
Therefore the following steps are to be performed:

1.  Make the following directories and ensure about 10 GB of free disk
    space: &lt;tmp&gt;/ and &lt;output&gt;/

2.  Start the container, depending on whether to evaluate on KITTI or
    DitM

    KITTI:

       ``` {.bash language="bash"} 
        sudo nvidia-docker run -ti -v <KITTIRoot>/:/home/data/KITTI/ \
        -v <ThesisRoot>/examples/:/home/mscnn/examples/ \
        -v <output>/:/home/output/ -v <tmp>/:/tmp/ caffe:MSCNN
       ```

    DitM:

       ``` {.bash language="bash"}
        sudo nvidia-docker run -ti -v <DitMRoot>/:/home/data/DitM \
        -v <ThesisRoot>/examples/:/home/mscnn/examples/ \
        -v  <output>/:/home/output/ -v <tmp>/:/tmp/ caffe:MSCNN
       ```

3.  Within the container go to examples/\{kitti\_car|DitM\}/ and run \lstinline[language=bash,breaklines]{python run_elementary_detection.py <FolderWithWantedNetwork> <FilenameOfWantedNetwork>}. If you want to do crossevaluation you have to adapt Line 61f within run\_elementary\_detection.py to the correct image folder!

4.  After the python script finished the output will have been saved to <output>/. Copy the archived results from output/ to <ThesisRoot>/examples/\{kitti\_car|DitM\}/ outputDetection/, depending on whether evaluations are done on KITTI or DitM

5.  Run evalOutputDetection within the right folder with MATLAB

6.  The final result (recall-precision over the validation set) will be saved to <ThesisRoot>/examples/\{kitti\_car|DitM\}/detections/<NameOfIntermediateResult>/plot/

[^1]: <http://www.cvlibs.net/download.php?file=data_object_image_2.zip>

[^2]: <http://www.cvlibs.net/download.php?file=data_object_label_2.zip>

[^3]: <http://droplab-files.engin.umich.edu/repro_200k_images.tgz>

[^4]: <https://goo.gl/2mpmw6>
