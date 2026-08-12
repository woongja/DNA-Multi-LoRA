# Table 4 -- EER (%) per noise domain on ASV19*
`baseline` is recomputed from raw scores; `noise_specific_lora` is quoted
from the paper.

| model        | method              | D0   |   D1 |    D2 |    D3 |    D4 |    D5 |   D6 |    D7 |
|:-------------|:--------------------|:-----|-----:|------:|------:|------:|------:|-----:|------:|
| AASIST-SSL   | baseline            | 0.15 | 3.67 |  3.86 | 15.37 | 10.42 | 22.87 | 8.75 | 40.01 |
| AASIST-SSL   | noise_specific_lora | -    | 2.30 |  0.70 |  1.79 |  0.92 |  4.19 | 3.57 |  4.38 |
| ConformerTCM | baseline            | 0.15 | 4.02 | 11.00 |  9.59 | 24.11 | 20.60 | 5.49 | 29.56 |
| ConformerTCM | noise_specific_lora | -    | 1.42 |  0.87 |  1.94 |  1.18 |  4.57 | 3.26 |  3.40 |
