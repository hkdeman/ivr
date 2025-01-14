# Introduction to Vision and Robotics
## Table of Contents
- [Installation guide](#installation-guide)	
- [How to run](#how-to-run)

## Installation guide

### Already have an installed simulator
If have already an installed simulator through the lab sessions, you only need to move some files across and install one additional package. To do this follow these steps:

1. Clone the assignment repo: `git clone https://github.com/ccimrie/IVR-assignment`
2. Change the name of your main.py within your IVR-Labs folder: `mv /path/to/folder/IVR-Labs/main.py /path/to/folder/IVR-Labs/main-labs.py`
3. Copy across the assignment files: `cp -r /path/to/assignment/IVR-assignment/reacher3D /path/to/folder/IVR-Labs && cp /path/to/assignment/IVR-assignment/main.py /path/to/folder/IVR-Labs`
4. Run the command: `WS=$PWD`
5. Clone the pyquaternion repo: `git clone https://github.com/KieranWynn/pyquaternion.git`
6. Go into this folder: `cd pyquaternion`
7. Install: `pip install . -I --prefix $WS/VRInstall`
8. Go to where main.py is: `cd $WS`

You should now have the assignmnet setup ready alongside your lab setup. You will still need to follow the same procedure on how to run as you did for the labs.

### Virtual Machine Installation 
If you are using Windows you can install a Virtual Machine (VM) that can run Ubuntu and from here you will be able to install easier the openai gym framework. The steps briefly are:

1. Download the correct installer: <https://www.virtualbox.org/wiki/Downloads>

2. Run the installer to install VirtualBox.

2. Download the Ubuntu 16.04 (desktop) ISO image: <https://www.ubuntu.com/download/alternative-downloads>

3. Start VirtualBox and click on new. Here you will then select the name of your VM and type; scroll down to Ubuntu.

4. After you will be asked to navigate to the ISO image file you downloaded in step 2.

5. One of the steps will be how much RAM and virtual hard disk space to assign. For now 1GB for RAM and 10GB for the hard disk should suffice (change as you feel is appropriate)

6. Here you will then install Ubuntu onto your VM. (When asked what to do with regards to the hard disk click use all, do not worry this is talking about the virtual hard disk). <https://tutorials.ubuntu.com/tutorial/tutorial-install-ubuntu-desktop-1604#0>

This is a brief set of steps and I recommend also looking at this link as it is a good reference: <https://medium.com/@tushar0618/install-ubuntu-16-04-lts-on-virtual-box-desktop-version-30dc6f1958d0>

### Initial steps for all machines
1. Install git: `sudo apt install git`
2. Delete the pip cache in the home directory: `rm -rf .cach/pip/`
3. Clone this repo: `git clone https://github.com/ccimrie/IVR-Labs.git`
4. cd to cloned folder: `cd IVR-Labs`

Follow the next steps according to your situation.

### Only on DICE machines
5. Run installation script: `source install.bash` :+1:

### Only on self-managed machines (Non-DICE or VM)

#### Without an existing Conda installation
5. Install python-pip: `sudo apt install python-pip`
6. Upgrade python-pip: `sudo pip install --upgrade pip`
7. Install required packages: `sudo apt install autotools-dev libtool automake autoconf mercurial pkg-config`
8. Run installation script: `source install.bash` :+1:

#### With an existing Conda installation
5. Create a new python environment: `conda create --name ivrlabenv python=2.7 pip scipy numpy cython -y`
6. Activate environment: `conda activate ivrlabenv`
7. Run installation script: `source conda_install.bash` :+1:

## How to run
On every new terminal you will have to navigate to the cloned folder and run `source setup.bash`.
This is to initialize the relevant environment variables.

If you want to avoid sourcing the setup.bash file with every new terminal, you can add it to you .bashrc file: `echo "source </path/to>/setup.bash" >> ~/.bashrc`.
Make sure you replace the `</path/to>` with the path to the setup file.

Also, in case you are using Conda, do not forget to activate the environment with `conda activate ivrlabenv`

# ivr
# ivr
