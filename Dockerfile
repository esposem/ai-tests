FROM quay.io/giantswarm/nvidia-gpu-toolkit

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

RUN pip3 install sentencepiece torch torchvision torchaudio transformers

ADD ./run_model.py /run_model.py

CMD python3 /run_model.py