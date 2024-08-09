Step 1: Install Python and Dependencies

Step 2: Install PyTorch for CPU

    pip install torch torchvision torchaudio

Step 3: Install Detectron2
  - By executing: pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html
  - Or by executing the following one after the other: 

        git clone https://github.com/facebookresearch/detectron2.git
        cd detectron2
        pip install -e .

  The second method installs Detectron2 in "editable" mode, allowing you to modify the source code if needed.

Step 4: Verify Installation
  
    import detectron2
    print(detectron2.__version__)
