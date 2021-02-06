import openpyxl

class Logger:

    def __init__(self, name):

        self.workbook = openpyxl.Workbook()

        self.name = name

        self.sheet = self.workbook.active
        self.sheet.title = str(self.name)

    def write_data(self, row, column, data):
        self.sheet.cell(row=row, column=column, value=data)

    def save_file(self):
        self.workbook.save(filename=str(self.name) + ".xlsx")