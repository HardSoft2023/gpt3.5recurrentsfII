



## FAQ
### 1、Preparing the Game Files
After the dependencies have been installed the necessary game files, all zipped inside of the StreetFighterIISpecialChampionEdition-Genesis directory, can be setup. The game files need to be copied into the actual game data files inside the installation of the retro library on your local machine. This location can be found by running the following lines in the command line:

python3
import retro
print(retro.__file__)

That should return the path to where the retro init.py script is stored. One level up from that should be the data folder. Inside there should be the stable folder. Copy the StreetFighterIISpecialChampionEdition-Genesis folder that is in the top level of the repo here. Inside the folder should be the following files:

-rom.md
-rom.sha
-scenario.json
-data.json
-metadata.json
-reward_script.lua
-Several .state files with each having the name of a specific fighter from the game

With that the game files should be correctly set up and you should be able to run a test agent.
### 2、How to visualize the game screen and the actions taken by the model during training on Mac OSX
I apologize for the confusion. It seems that `xvfb` is not available as a standalone package on macOS. Instead, you can use Xvfb as part of the X11 windowing system that is included with XQuartz.

Here are the steps to use Xvfb with XQuartz on macOS:

1. Install XQuartz from the official website: https://www.xquartz.org/

2. Open XQuartz and go to the XQuartz menu -> Preferences.

3. In the Preferences window, go to the "Security" tab and ensure that the "Allow connections from network clients" checkbox is selected.

4. Close the Preferences window and open a new terminal window.

5. In the terminal window, run the following command to start Xvfb:

```shell
Xvfb :1 -screen 0 1400x900x24 &
```

6. Set the `DISPLAY` environment variable to `:1`:

```shell
export DISPLAY=:1
```

7. Run your `train.py` script as before, but without `xvfb-run`:

```shell
python train.py
```

This should start the virtual framebuffer and allow you to run the `train.py` script with the game screen rendered without an actual physical display.

## References
* https://github.com/corbosiny/StreetFighterAI.git
* https://github.com/linyiLYi/street-fighter-ai.git
* [格斗之王！AI写出来的AI竟然这么强](https://www.bilibili.com/video/BV1DT411H7ph/?share_source=copy_web&vd_source=46d64ecd4995954948ea0cf688f2ba30)