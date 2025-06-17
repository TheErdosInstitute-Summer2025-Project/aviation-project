Run time: 17/06/2025 18:25:51

Json query: {"ExportFormat":"summary","ResultSetSize":500,"ResultSetOffset":1500,"TargetCollection":"cases","AndOr":"And","QueryGroups":[{"AndOr":"And","QueryRules":[{"RuleType":0,"Values":["Aviation"],"Columns":["Event.Mode"],"Operator":"contains"},{"RuleType":0,"Values":["2022-01-01","2022-12-31"],"Columns":["Event.EventDate"],"Operator":"is in the range"},{"RuleType":0,"Values":["USA"],"Columns":["Event.Country"],"Operator":"is"}]}],"SortColumn":null,"SortDescending":true,"SessionId":211257}

Query text: [
	(Mode) contains (Aviation)
	And
	(EventDate) is in the range (2022-01-01 OR 2022-12-31)
	And
	(Country) is (USA)
]

