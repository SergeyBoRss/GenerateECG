return_code=0

echo "flake8"
flake8 ecg random_ecg.py
return_code=$(($return_code | $?))

echo "mypy"
mypy ecg random_ecg.py
return_code=$(($return_code | $?))

echo "isort"
isort ecg random_ecg.py --diff --check-only
return_code=$(($return_code | $?))

exit $return_code
