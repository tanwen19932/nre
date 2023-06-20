# FROM jianwei.conda-gpu-base:1.0
FROM datajoint/miniconda3:22.11.1-py3.7-debian
MAINTAINER tanwen

ARG RELEASE

ARG LAUNCHPAD_BUILD_ARCH
#如果是GPU 需要设置如下内容才可以进行执行，如果是复制构建镜像方案
LABEL org.opencontainers.image.ref.name=ubuntu
LABEL org.opencontainers.image.version=18.04
LABEL com.nvidia.volumes.needed=nvidia_driver
ENV PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PYTORCH_VERSION=v2.0.0


ARG CONDA_ENV=py37
# 如果需要更换python版本等，注意老版本需要删除base
COPY ./install/docker/base-gpu/condarc ~/.condarc
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
RUN conda config --set ssl_verify false
RUN conda create -n py37 python=3.7


#如果使用conda并有环境
ENV PATH /opt/conda/envs/${CONDA_ENV}/bin:$PATH
ENV CONDA_DEFAULT_ENV ${CONDA_ENV}
WORKDIR /workspace
COPY . /workspace
RUN pip install --upgrade pip -i http://pypi.douban.com/simple --trusted-host pypi.douban.com && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install -r requirements.txt && \
    rm -rf ~/.cache

ENV LANG="C.UTF-8"
USER root
# conda环境默认激活，root权限可以执行命令 ，增加pythonpath执行方便各个workingdir都能够找到对应的module
RUN echo 'export PYTHONPATH="${PYTHONPATH}:/workspace"' >> ~/.bashrc
RUN echo  'export LANG="C.UTF-8"' >> ~/.bashrc
RUN echo  'conda activate py37' >> ~/.bashrc
ENV PYTHONPATH="${PYTHONPATH}:/workspace"

WORKDIR /workspace/tw_word2vec
# 如果能保证镜像可以用 这里CMD输入相关启动命令即可，或是docker执行的时候手动指明即可，建议docker执行手动指明，因为有训练服务等不同功能
ENTRYPOINT ["/bin/bash"]
CMD ["-c","python sem_eval_08.py"]