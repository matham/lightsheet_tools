# Installation

## Fiji

Install `fiji` from [here](https://imagej.net/software/fiji/downloads) as instructed.

Open Fiji and increase the total RAM available and the number of threads as appropriate for the machine. Go to ``.

### BigStitcher

In `Fiji` install the `BigStitcher` plugin [following their instruction](https://imagej.net/plugins/bigstitcher/#download). 

Overall, in the menu go to `Help › Update…`, click `Manage update sites` and select `BigStitcher` in the list. After applying the changes and restarting Fiji, `BigStitcher` will be available under `Plugins › BigStitcher › BigStitcher`.

Next make sure BigStitcher has a bug fixed version for our images as described [here](https://github.com/PreibischLab/BigStitcher/issues/134#issuecomment-1876161165).
If a fix wasn't merged to their repo, get a manually compiled `multiview-reconstruction-xxx.jar`
and copy it into `Fiji.app\plugins` deleting the original version of that file (see next section).

#### Compiling multiview-reconstruction

##### Setup

Install [IntelliJ IDEA Community Edition](https://www.jetbrains.com/idea/). Got to download page and scroll down
to download the community edition.

Ensure all the plugins in Fiji is updated to the latest version (`help -> update`). Download the latest
tag of [multiview-reconstruction](https://github.com/PreibischLab/multiview-reconstruction/tags). E.g.
[multiview-reconstruction-5.4.0](https://github.com/PreibischLab/multiview-reconstruction/archive/refs/tags/multiview-reconstruction-5.4.0.zip)
and extract to e.g the `PycharmProjects` directory. 
Download the zip [ZuluJDK8+FX](https://www.azul.com/downloads/?version=java-8-lts&os=windows&architecture=x86-64-bit&package=jdk-fx#zulu)
and extract it to a subdirectory of the project (or somewhere permanent).

In the IDE, create new project, browse to the directory of `multiview-reconstruction` and name the project
the name of the root directory (e.g. `multiview-reconstruction-5.4.0`). Select Maven as the build system. For
the JDK select `add from disk` and browse to the root JDK directory you previously extracted. Disable
`Add sample code`. And hit create.

The IDE likely overwrites the `pom.xml` file after creating the project. Get the `pom.xml` file from the
zip file and overwrite the one in the extracted project.

##### Build

In the IDE open file `src/main/java/net/preibisch/mvrecon/process/fusion/FusionTools.java`. Around line 118
locate `public static float defaultBlendingRange = 40;`. Change 40 to a large number, just larger than the maximum
expected overlap. E.g. `1000`.

Find the Maven icon, select the `Multiview Reconstruction` project, and click the build icon. Once built, locate the
built jar, e.g. `target/multiview-reconstruction-5.4.0.jar`. In `Fiji.app/plugins` find the existing plugin,
e.g. `multiview-reconstruction-5.4.0.jar` rename it to e.g. `multiview-reconstruction-5.4.0.jar~original` and
copy the built plugin into the directory instead. Now, when opened, Fiji should load our version.

## Python

Create the Python environment used for running stitching in Fiji.

- Install PyImage in mamba as described [here](https://py.imagej.net/en/latest/Install.html#installing-via-conda-mamba). Overall get mamba via [miniforge](https://github.com/conda-forge/miniforge#miniforge3) and do:
  ```shell
  conda config --add channels conda-forge
  conda config --set channel_priority strict
  conda create -n pyimagej pyimagej openjdk=11
  ```
- Install the remaining dependencies in the created conda pyimagej environment:
  
  ```shell
  conda install ome-types xsdata psutil
  pip install tqdm
  ```

## Imaris

Install [Imaris Viewer](https://imaris.oxinst.com/imaris-viewer) and
[Imaris File Converter](https://imaris.oxinst.com/microscopy-imaging-software-free-trial) from their
website - it's free.

# Script Setup

In the `stitch.py` script update
- `fiji_path` to point to the full `Fiji.app` path e.g. `C:\Users\CPLab\Fiji.app`. 
- `imaris_converter` to the fill Imaris converter exe, e.g.
  `C:\Program Files\Bitplane\ImarisFileConverter 10.2.0\ImarisConvert.exe`.
- Update the `tiff_drive` to the directory that contains the input tiff files.
  Something like `E:\imaging`. This folder should contain a `staging` folder
  that contains your data. The data should be organized as a sub-folder for each date
  (e.g. `20250225`), that itself contains sub-folders of your tiff files. Each sub-folder
  will be e.g. `250225_MF1_138F_W_L_16-01-47` that contains the tiles as tiff files for each
  tile and channel. If using a right and left laser channel, each channel will have its own folder
  with its tiff files.
- Update the `other_drives` to point to all the available drives on the computer to be used as intermediate drives.
- Update the `copy_raw_paths`, `copy_fused_paths`, and `copy_ims_path` to the appropriate directories where
  the raw / stitched data should saved to.
- Update any additional parameters such as number of tiles and gear factor.
