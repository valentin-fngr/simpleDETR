mkdir coco
cd coco
mkdir images
cd images

wget http://images.cocodataset.org/zips/test2017.zip

unzip test2017.zip

rm test2017.zip

cd ../
wget http://images.cocodataset.org/annotations/image_info_test2017.zip

unzip image_info_test2017.zip

rm image_info_test2017.zip
