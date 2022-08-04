
sudo docker run --gpus all -it  \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-e DISPLAY=unix$DISPLAY \
-e GDK_SCALE \
-e GDK_DPI_SCALE \
-v /home/u20:/home/u20 \
-v /home/u20/.pip:/root/.pip \
--name=cu13 nvidia/cuda:10.2-devel bash 

sudo docker run --gpus all -it  \
--rm \
nvidia/cuda:10.2-devel nvidia-smi 