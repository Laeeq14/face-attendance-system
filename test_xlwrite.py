# test_xlwrite.py
import xlwrite

# Test data
filename = 'test_attendance'
sheet_name = 'class1'
row = 1
id = 1
name = 'John Doe'
status = 'Present'

# Test function call
filepath = xlwrite.output(filename, sheet_name, row, id, name, status)
print(f"Data written to {filepath}")
