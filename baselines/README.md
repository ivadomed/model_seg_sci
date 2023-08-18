### Install ANIMA

ANIMA is used to compute segmentation performance metrics.

```
##### STEP 1: Install ANIMA #####
cd ~
mkdir anima/
cd anima/
wget -q https://github.com/Inria-Visages/Anima-Public/releases/download/v4.2/Anima-Ubuntu-4.2.zip   (change version to latest)
unzip Anima-Ubuntu-4.2.zip
git lfs install
git clone --depth 1 https://github.com/Inria-Visages/Anima-Scripts-Public.git
git clone --depth 1 https://github.com/Inria-Visages/Anima-Scripts-Data-Public.git


##### STEP 2: Configure directories #####
cd ~
mkdir .anima/
touch .anima/config.txt

echo "[anima-scripts]" >> .anima/config.txt
echo "anima = ${HOME}/anima/Anima-Binaries-4.2/" >> .anima/config.txt
echo "anima-scripts-public-root = ${HOME}/anima/Anima-Scripts-Public/" >> .anima/config.txt
echo "extra-data-root = ${HOME}/anima/Anima-Scripts-Data-Public/" >> .anima/config.txt
```