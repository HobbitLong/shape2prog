# download the model
mkdir -p model

cd model

wget http://shape2prog.csail.mit.edu/repo/model.tar.gz -O model.tar.gz
tar zxvf model.tar.gz
rm model.tar.gz

cd ..

# download the data
mkdir -p data

cd data

wget wget http://shape2prog.csail.mit.edu/repo/data.tar.gz -O data.tar.gz
tar zxvf data.tar.gz
rm data.tar.gz

cd ..
