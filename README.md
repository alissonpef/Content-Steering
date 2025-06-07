## Content Steering: Leveraging the Computing Continuum to Support Adaptive Video Streaming 

This repository was created to host the materials SBRC shortcourse.

### Abstract

Video streaming has become one of the most used Internet applications nowadays, with numerous leading technology companies competing for dominance in a market valued in the billions. The delivery of high-quality streaming services necessitates strategic utilization of computing resources near end-users, with emerging technologies like 6G and edge-cloud continuum infrastructures being explored to meet these growing demands. These technologies promise to enable rapid, reliable data transfer for large data volumes, with the edge-cloud continuum facilitating service placement mobility from central cloud data centers to edge devices near users. However, managing seamless service mobility and precise computing resource allocation for quality service remains complex. The zero-touch network concept, eliminating the need for manual network configuration, is becoming popular in this context. Specifically, in video streaming, the integration of the Content Steering architecture from the Dynamic Adaptive Streaming over HTTP (DASH) protocol with container orchestrator technologies could allow for autonomous video streaming service placement across the continuum, reducing human involvement and optimizing computing resource use. Our short course provides a hands-on experience with the latest technology in this domain, teaching participants about cutting-edge architectures and tools for creating and managing adaptive video streaming applications using the latest content steering architecture introduced in the DASH protocol. Participants will build a small edge-cloud virtual and local testbed to explore request steering strategies for video content across the computing continuum. The course also addresses current challenges and future research opportunities in this evolving field.


## Prerequisites:

- [mkcert](https://github.com/FiloSottile/mkcert)
- [docker](https://www.docker.com/)

## VirtualBox VM:

We provide a fully configured VirtualBox VM with all the necessary software for you to download and explore our tutorial's testbed. You can download the pre-configured VM in the following link: https://drive.google.com/file/d/1mCB585muebdJIN6yXbioIoD1762svy3T/view?usp=sharing

To use the VM, you'll need VirtualBox installed on your system. For more information on how to install VirtualBox or how to import a VM image, please refer to: https://www.virtualbox.org/

### Executing the video streaming on the virtual box environment

1. After starting the machine, access the folder "Documents/content-steering-tutorial" and execute the starting-streaming.sh script ($ ./starting-streaming.sh)
2. Verify whether the docker containers started ($ docker ps)
3. Get information on the IP address of each cache node ($ docker inspect video-streaming-cache-1)
4. For each cache node inspected, add a line in the /etc/hosts file following the pattern: 172.18.0.2  video-streaming-cache-1.

```
172.18.0.2	video-streaming-cache-1
172.18.0.4	video-streaming-cache-2
172.18.0.3	video-streaming-cache-3
```

5. Now the edge nodes are up and running, we need to start the content steering orchestrator. In the "Documents/content-steering-tutorial" type ($ python3 steering-service/src/app.py)
6. Open Google Chrome and go to the video player GUI: https://reference.dashif.org/dash.js/latest/samples/advanced/content-steering.html
7. In the URL, access: https://video-streaming-cache-1/Eldorado/4sec/avc/manifest.mpd

## Configure Environment Tutorial:


## 1. Clone the Tutorial Repository
First, clone the tutorial repository from GitHub using Git:

```shell
git clone https://github.com/robertovrf/content-steering-tutorial
```

## 2. Set Up Local Streaming Service

### 2.1 Set Up Local Custom Domains and Certificates 
Edit your local hosts file located at /etc/hosts to assign local custom domain names for streaming and steering services. Then, run the script create_certs.sh to generate certificates and enable HTTPS in localhost:

```shell
./create_certs.sh <streaming-domain> <steering-domain>
```

### 2.1 Download DASH Video Dataset

Download a DASH video from the mmsys dataset and save it to the designated folder named dataset. The dataset offers various codec options such as AV1, AVC, HEVC, and VVC. You can use wget or any other method to download the video.


### 2.2 Start Local Streaming Service

Run the script create_streaming.sh to initiate the regional streaming setup where the streaming services will run:

```shell
./starting_streaming.sh
```

You can now verify if the local video streaming is operational. Access the [dash.js](https://reference.dashif.org/dash.js/latest/samples/dash-if-reference-player/index.html) player and attempt to load the manifest using the following URL format:

```shell
https://streaming-service/<streaming path>/manifest.mpd
```

Replace <streaming-service> with your streaming domain and <streaming-path> with the path to your video.


## References

- [mmsys22 Dataset](https://doi.org/10.1145/3524273.3532889) 
Babak Taraghi, Hadi Amirpour, and Christian Timmerer. 2022. Multi-codec ultra high definition 8K MPEG-DASH dataset. In Proceedings of the 13th ACM Multimedia Systems Conference (MMSys '22). Association for Computing Machinery, New York, NY, USA, 216–220. 
