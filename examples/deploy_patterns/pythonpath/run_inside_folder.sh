XYZ=$(pwd)
echo $XYZ
cd workflows
echo $(pwd)
PYTHONPATH=${XYZ} python workflow.py
