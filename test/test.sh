cd a1
python test.py
cd ../a2
python test.py
cd ../a3
python test.py
cd ../a4
python test.py
cd ../b1
python test.py
cd ../b2
python test.py
cd ../b3
python test.py
cd ..
python combine.py
cd submit
zip -r submit.zip .
cd ../submit2
zip -r submit2.zip .
