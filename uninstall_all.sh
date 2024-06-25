while read -r package; do
    pip uninstall -y "$package"
done < requirements.txt
