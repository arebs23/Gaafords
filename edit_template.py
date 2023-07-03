

environment  = 'tabletop'

query_template =  f"""Imagine {labels} on a {environment} 
 What are the possible interactions between these objects? Specify answer by giving names of Action, object performed on, instrument used to perform the action and the effect on the object. 
 An example is:
   ###Action:Peel
   Performed on: Onion 
   Instrument: knife
   Effect: peeled(onion)### """