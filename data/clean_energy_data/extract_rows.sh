grep 'INSERT INTO `data_calcqcset1`' cepdb_2013-06-21.sql | awk -v file=ids.txt '
BEGIN {
    while ((getline line < file) > 0) {
        ids[line] = 1;
    }
}
{
    insert_stmt = "INSERT INTO `data_calcqcset1` VALUES ";
    start = index($0, "(");
    if (start > 0) {
        row_data = substr($0, start);  # Extract the actual row values
	split(row_data, rows, /\),\(/); # Split all rows
        first = 1;
        for (i in rows) {
            match(rows[i], /^[(]?[0-9]+,([0-9]+),/, arr);
            if (arr[1] in ids) {
                if (first) {
                    printf "%s(%s)\n", insert_stmt, rows[i] >> "filtered_data_calcqcset1.sql";
                    first = 0;
                } else {
                    printf ",(%s)\n", rows[i] >> "filtered_cepdb_2013-06-21.sql";
                }
            }
        }
    }
}' 

