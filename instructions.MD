## Instructions to update the puffer source code
In order to save/print the model output for each video chunk, change the source code ant then run the following commands:
```
cd ~/puffer/
make clean
./autogen.sh
./configure
make -j CXXFLAGS='-DNONSECURE'
```
