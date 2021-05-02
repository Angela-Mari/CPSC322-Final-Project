import mysklearn.myutils as myutils
import copy
import csv 
from tabulate import tabulate 

# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.
    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data. There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.
        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).
        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """

        M = len(self.column_names)
        N = len(self.data)
        return [N, M]

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.
        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.
        Returns:
            list of obj: 1D list of values in the column
        Notes:
            Raise ValueError on invalid col_identifier
        """
        try:
            col_index = self.column_names.index(col_identifier)
        except ValueError:
            return []

        if include_missing_values:
            col = []
            for row in self.data: 
                col.append(row[col_index])
        else:
            col = []
            for row in self.data: 
                if row[col_index] != "NA":
                    col.append(row[col_index])
        
        return col
    
    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).
        Notes:
            Leave values as is that cannot be converted to numeric.
        """

        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                try:
                    self.data[i][j] = float(self.data[i][j])
                except ValueError:
                    continue

    def drop_rows(self, rows_to_drop):
        """Remove rows from the table data.
        Args:
            rows_to_drop(list of list of obj): list of rows to remove from the table data.
        """
        new_data = []
        for i in range(len(self.data)):
            if self.data[i] in rows_to_drop:
                continue
            else:
                new_data.append(self.data[i])
        
        self.data = new_data

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.
        Args:
            filename(str): relative path for the CSV file to open and load the contents of.
        Returns:
            MyPyTable: return self so the caller can write code like: table = MyPyTable().load_from_file(fname)
        
        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        
        table = []
        header = []

        infile = open(filename, "r")
        lines = infile.readlines()
        
        for line in csv.reader(lines):
            table.append(line)

        infile.close

        for i in range(len(table[0])):
            header.append(table[0][i])
        
        table = table[1:]
        
        self.__init__(header,table)
        self.convert_to_numeric()

        return self 

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.
        Args:
            filename(str): relative path for the CSV file to save the contents to.
        Notes:
            Use the csv module.
        """
        outfile = open(filename, "w")
        
        # col header
        for i in range(len(self.column_names) -1):
            outfile.write(self.column_names[i] + ",")
        outfile.write(self.column_names[-1] + "\n")
        
       
        # write data
        for row in (self.data):
            for i in range(len(row) - 1):
                outfile.write(str(row[i]) + ",")
            # get rid of extra line but not requiered 
            # if (row == self.data[-1]):
            #     outfile.write(str(row[i + 1]))
            # else: 
            outfile.write(str(row[i + 1]) + "\n")
        
        outfile.close()

    def find_duplicates(self, key_column_names):
        """Returns a list of duplicates. Rows are identified uniquely based on key_column_names.
        Args:
            key_column_names(list of str): column names to use as row keys.
        Returns: 
            list of list of obj: list of duplicate rows found
        
        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s). The first instance is not
                considered a duplicate.
        """
        indexs = []

        for item in key_column_names:
            indexs.append(self.column_names.index(item))

        unique_keys = []
        duplicate_rows = []
        for i in range(len(self.data)):
            my_key = []
            for j in range(len(indexs)):
                my_key.append(self.data[i][indexs[j]])
            if my_key in unique_keys:
                duplicate_rows.append(self.data[i])
            else:
                unique_keys.append(my_key)

        return duplicate_rows

    def find_keys(self, key_column_names):
        indexs = []

        for item in key_column_names:
            indexs.append(self.column_names.index(item))

        keys = []
        #duplicate_rows = []
        for i in range(len(self.data)):
            my_key = []
            for j in range(len(indexs)):
                my_key.append(self.data[i][indexs[j]])
            if my_key in keys:
                continue
            else:
                keys.append(my_key)

        return keys

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        remove = []
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                if self.data[i][j] == "NA" or self.data[i][j] == "":
                    remove.append(self.data[i])
        
        self.drop_rows(remove)

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column by the column's original average.
        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """

        try:
            col_index = self.column_names.index(col_name)
        except ValueError:
            return []

        total = 0
        length = 0
        rows = []

        for i in range(len(self.data)):
            if self.data[i][col_index] == "NA" or self.data[i][col_index] == "":
                rows.append(i)
            else:
                total = total + self.data[i][col_index]
                length = length + 1
        
        avg = total/length

        for i in range(len(rows)):
            self.data[rows[i]][col_index] = avg

    def get_total(self, col):
        """Calculates total of continuous list
        Args:
            col(list of int or float): to add together
        Returns:
            total: total for the column 
        """
        total = 0
        for item in col:
                total = total + item
        return total

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.
        Returns:
            MyPyTable: stores the summary stats computed.
        """
        # these are the col headers
        header = ["attribute", "min", "max", "mid", "average", "median"]

        summary_data = []

        # calculate for each col passed in and append
        # sort col first

        for item in col_names:
            
            min = 0 
            max = 0 
            mid = 0
            avg = 0 
            median = 0 
            row = []
            row.append(item)
            
            col = self.get_column(item, False)

            # edge case
            if len(col) == 0:
                return MyPyTable(header, [])

            col.sort()
            

            min = col[0]
            row.append(min)
            
            max = col[-1]
            row.append(max)
            
            mid = (min + max) / 2 
            row.append(mid)

            total = self.get_total(col)
            avg = total/len(col)
            row.append(avg)

            if (len(col)%2 == 0): #even
                higher = len(col)//2
                lower = higher - 1
                median = (col[higher]+col[lower])/2
            else:
                median = col[len(col)//2]
            row.append(median)

            summary_data.append(row)

        return MyPyTable(header, summary_data) 

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined with other_table based on key_column_names.
        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.
        Returns:
            MyPyTable: the inner joined table.
        """
        
        
        
        # you will have to search self table to find "original"
        # then combine the data into join table
        self.pretty_print()
        other_table.pretty_print()

        # create header
        header = []
        for item in self.column_names:
            header.append(item)
        
        for item in other_table.column_names:
            if item not in header:
                header.append(item)

        join_data = []
        
        # create rows out of keys
        key_indecies = []
        for item in key_column_names:
            key_indecies.append(self.column_names.index(item))

        key_rows = []
        for row in self.data:
            search_key = []
            for index in key_indecies:
                search_key.append(row[index])
            for other_row in other_table.data:
                if(all(x in other_row for x in search_key)):
                    key_rows.append(search_key)
       
        join_rows = []
        for item in key_rows:
            if item not in join_rows:
                join_rows.append(item)

        # then combine the data into join_data
        for row in join_rows:
            
            for self_row in self.data:
                join_row = []
                if(all(x in self_row for x in row)): 
                    join_row.append(self_row)
                    for other_row in other_table.data:
                        if(all(x in other_row for x in row)):
                            for item in other_row:
                                if item not in row:
                                    if item not in join_row[0]:
                                        join_row[0].append(item)
                if join_row != []:
                    join_data.append(join_row[0])

        join_table = MyPyTable(header, join_data)
        join_table.pretty_print()

        return MyPyTable(header, join_data)

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with other_table based on key_column_names.
        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.
        Returns:
            MyPyTable: the fully outer joined table.
        Notes:
            Pad the attributes with missing values with "NA".
        """

        # create header
        header = []
        for item in self.column_names:
            header.append(item)
        
        for item in other_table.column_names:
            if item not in header:
                header.append(item)
        
        join_data = copy.deepcopy(self.data)
        
        key_indecies = []
        for item in key_column_names:
            key_indecies.append(other_table.column_names.index(item))

        for row in other_table.data:
            found = False
            search_key = []
            for key in key_indecies:
                search_key.append(row[key])
            for join_row in join_data:
                if(all(x in join_row for x in search_key)):
                    for item in row:
                        if item not in search_key:
                            join_row.append(item)
                            found = True
            if not found:
                new_row = []
                for i in range(len(self.column_names)):
                    new_row.append("NA")
                new_row_key_indecies = []
                for key in key_column_names:
                    new_row_key_indecies.append(header.index(key))
                for i in range(len(search_key)):
                    new_row[new_row_key_indecies[i]] = search_key[i]
                for item in row:
                    if item not in new_row:
                        new_row.append(item)


                join_data.append(new_row)

        #check for missing NA
        for row in join_data:
            if len(row) < len(header):
                row.append("NA")

        return MyPyTable(header,join_data) 
