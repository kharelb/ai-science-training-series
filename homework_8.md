- **How could you use AI for a problem that interests you?**
  
  `I am a radio astronomy student and I am working in Fast Radio Bursts(FRBs).
   The FRBs are millisecond bursts of extragalactic origin.
  There are very few tools that are used to model the FRBs and we worked and published
   one such tool to model the FRBs and radio pulsars. Modeling
  FRBs are important because it gives us the nature of the burst and
   it's underlying physical properties. With the advancement of technology
  more and more FRBs are detected everyday but modeling FRBs using
   traditional techniques is very time consuming and needs lot of human intervention.
  I would like to use an AI tool to model FRBs that allows more robust and quick modeling
  and with very less to no human interventions.`

- **What is the task?**
  
  `The task is to model the Fast Radio Bursts.`

- **What kind of data would you use?**
  
  `The data I would use are 2D numpy array with dim-0 axis representing time and dim-1 representing
  frequency of observation. Each numpy array contains a radio signal, which can be viewd by using
  some 2D plotting tool.`

- **What kind of method or model might be appropriate?**
  
  `I think CNN with transformers can be useful. I am also thinking about if LLMs can be used too.`

- **What kind of metric would you use to measure success?**
  
  `After modeling we can recreate a signal with the model parameters. We can calculate chi-squared loss to evaluate the
  model thereafter. `
