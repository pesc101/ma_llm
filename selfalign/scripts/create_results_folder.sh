## ------------------------------------------- Create Folder -------------------------------------------------------


create_folder() {
    # Get the current date in the format YYYY-MM-DD
    current_date=$(date +%Y-%m-%d)

    # Check if the folder already exists
    if [ -d "selfalign/results/aug/${current_date}" ]; then
        # Folder already exists, find the next available number
        count=1
        while [ -d "selfalign/results/aug/${current_date}_$count" ]; do
            ((count++))
        done

        # Append the count to the current date
        current_date="${current_date}_$count"
    fi

    # Create the folder with the new name
    final_folder="selfalign/results/aug/$current_date"
    mkdir "$final_folder"
    echo "Folder $final_folder created."
}
