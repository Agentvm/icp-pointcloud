# Ubuntu
```
python --version
>Python 2.7.12
python3 --version
>Python 3.5.2
```
```
python -m pip install --user -U pip setuptools
python3 -m pip install --user -U pip setuptools
```

```
python3 -m pip install --user python-language-server[all]=0.19.0
```

```
python -m pip install --user matplotlib numpy scikit-learn cython opencv-python
python3 -m pip install --user matplotlib numpy scikit-learn cython opencv-python
```

## python-pcl

Follow https://github.com/strawlab/python-pcl/tree/master

and then checkout the repository itself, cd  inside it and do

```
sudo python setup.py install
sudo python3 setup.py install
```

You may need up to 10 GB free space on your Ubuntu drive.

# Windows

## python

#### Download

This https://python-pcl-fork.readthedocs.io/en/rc_patches4/install.html#install-python-pcl
says: The following versions of Python can be used: 2.7.6+, 3.5.1+, and 3.6.0+.

python-pcl is supported on Python 2.7.6+, 3.4.0, 3.5.0+, 3.6.0+.

so lets pick 2.7.newest and 3.6.newest. From https://www.python.org/, download the respective files and install them.

#### Path Variable

On your Keyboard, press the Windows Button and type 'environment variables', or 'umgebungsvariablen'. You want to
select the option that lets you change the system variables, not the user variables. Edit the Path Variable and add the following:

###### Windows 10
`pathToYourPython2Installation\`

`pathToYourPython2Installation\Scripts\`

`pathToYourPython3Installation\`

`pathToYourPython3Installation\Scripts\`


###### Windows 8
`;pathToYourPython2Installation\;pathToYourPython2Installation\Scripts;pathToYourPython3Installation\;pathToYourPython3Installation\Scripts`

#### Getting along
This way, Python2 will be found first. For Python3 to be found at all, go tot `pathToYourPython3Installation\` and COPY the python.exe. Then
rename THE COPY to python3.exe. The Modules still need to reference a python.exe. But things are different in terminal.

#### Testing
Open a terminal by pressing Windows + R and typing cmd or by pressing Windows and searching for cmd (this way you could add admin authority).
Try these commands, they should give the according output:

```
python --version
>>>Python 2.7.12

python3 --version
>>>Python 3.4.4
```


#### Pip and Setuptools
If all is well, start updating.

```
python -m pip install -U pip setuptools
python3 -m pip install -U pip setuptools
```

#### Python Language Server
And installing.

```
python3 -m pip install python-language-server[all]==0.19.0
```

Version 0.19.0 seemed to be the last that works. This Enables the ide-python atom package to work.
At this point, you should be able to debug and execute any python file in [Atom](#atom).

#### Miscellaneous
```
python -m pip install matplotlib numpy scikit-learn cython opencv-python
python3 -m pip install matplotlib numpy scikit-learn cython opencv-python
```

#### python-pcl
Following https://github.com/strawlab/python-pcl/tree/master


# Atom

## package list
-	higlight selected
-	minimap
-	minimap-highlight-selected
-	atom-ide-ui
-	process-palette
-	ide-python
-	autocomplete-python

install packages by opening setting with "Ctrl + ;", then selet the tab Install. Search for all the listed packages and install them.

## process-palette configuration

In the topmost bar of atom, select Packages>Process Palette>Toggle. Press the first Button that says "Do It!" and find the
Button with the pencil icon labelled "Edit configuration". Choose the global configuration. Delete all existing configurations
and make a new one. Fill it as follows

### Python2

#### Action Name
`python`

#### Shell Command
```
python {filePath}
```
use the Insert Variable Button, if preferred

#### Working Directory
```
{fileProjectPath}
```

#### Keystroke
```
f5
```

#### Saving
Select 'Referenced' and uncheck 'Prompt before saving'

### Python3
Make a new Configuration and change the Action Name and the Shell Command to
```
python3
```
change the Keystroke also.

### test it
Open a new file, save it as .py and paste the following inside:
```
import sys

print ('executed with python version ' + str (sys.version_info[0] ) + '.' + str(sys.version_info[1]) + "\n" )
```
Press your designated keystroke.
Voilá. What went wrong?



## ide python configuration

### Windows

Set Option "Python Executable" to the full path of your python3 executable (ending in .exe). Then Restart Atom.

### Ubuntu

Set Option "Python Executable" to "python3". Then Restart Atom.

(Optional) While programming, keep these options open and add to "PyCodeStyle>Ignore", for example E211, W391, E202, E265
