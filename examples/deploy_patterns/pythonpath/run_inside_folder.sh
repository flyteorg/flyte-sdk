XYZ=$(pwd)
echo $XYZ
cd workflows
PYTHONPATH=${XYZ} python workflow.py