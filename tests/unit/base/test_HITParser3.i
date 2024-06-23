[Models]
  [foo]
    type = SampleParserTestingModel
    Real_seq = '1.1   2.2 -3.3 '
    Real_seq_from_csv_col_name = 'unit/base/test.csv:foo'
    Real_seq_from_csv_col_index = 'unit/base/test.csv:[1]'
    string_seq = 'today is a good day '
    string_seq_from_csv_col_name = 'unit/base/test.csv:baz'
    string_seq_from_csv_col_index = 'unit/base/test.csv:[3]'
  []
[]
