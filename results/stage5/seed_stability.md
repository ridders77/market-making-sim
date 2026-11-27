# Seed stability and action use

Share of states (visited ≥ 50 times on average) where the 5 seeds' greedy actions agree.

        variant side    fee  states  all_5_agree  4plus_agree
        private  buy retail      25         0.52         0.76
        private  buy    pro      25         0.52         0.88
        private sell retail      25         0.56         0.84
        private sell    pro      25         0.56         0.88
            ofi  buy retail      75         0.31         0.61
            ofi  buy    pro      75         0.33         0.60
            ofi sell retail      75         0.35         0.63
            ofi sell    pro      75         0.40         0.76
          queue  buy retail      75         0.29         0.53
          queue  buy    pro      75         0.28         0.53
          queue sell retail      75         0.27         0.63
          queue sell    pro      75         0.40         0.63
          sigma  buy retail      75         0.43         0.67
          sigma  buy    pro      75         0.45         0.76
          sigma sell retail      75         0.39         0.60
          sigma sell    pro      75         0.45         0.75
ofi+queue+sigma  buy retail     675         0.02         0.11
ofi+queue+sigma  buy    pro     675         0.11         0.32
ofi+queue+sigma sell retail     675         0.01         0.12
ofi+queue+sigma sell    pro     675         0.08         0.28

Test-day decisions: share of episodes whose first action is cross / touch / behind, share with any cross, and the share of decisions that used the unvisited-state fallback.

side    fee         variant  first_cross  first_touch  first_behind  any_cross  fallback_share
 buy retail         private          0.0     1.000000      0.000000   0.000000             0.0
 buy retail             ofi          0.0     0.732163      0.267837   0.000116             0.0
 buy retail           queue          0.0     0.673149      0.326851   0.000493             0.0
 buy retail           sigma          0.0     0.960987      0.039013   0.000697             0.0
 buy retail ofi+queue+sigma          0.0     0.539594      0.460406   0.032743             0.0
 buy    pro         private          0.0     0.800000      0.200000   0.000000             0.0
 buy    pro             ofi          0.0     0.804151      0.195849   0.000000             0.0
 buy    pro           queue          0.0     0.663425      0.336575   0.008331             0.0
 buy    pro           sigma          0.0     0.960987      0.039013   0.000232             0.0
 buy    pro ofi+queue+sigma          0.0     0.653672      0.346328   0.025283             0.0
sell retail         private          0.0     0.600000      0.400000   0.000000             0.0
sell retail             ofi          0.0     0.799361      0.200639   0.000000             0.0
sell retail           queue          0.0     0.589434      0.410566   0.000174             0.0
sell retail           sigma          0.0     0.654514      0.345486   0.000087             0.0
sell retail ofi+queue+sigma          0.0     0.607721      0.392279   0.032569             0.0
sell    pro         private          0.0     1.000000      0.000000   0.000000             0.0
sell    pro             ofi          0.0     0.804151      0.195849   0.000000             0.0
sell    pro           queue          0.0     0.660987      0.339013   0.007315             0.0
sell    pro           sigma          0.0     0.846763      0.153237   0.000000             0.0
sell    pro ofi+queue+sigma          0.0     0.664993      0.335007   0.017126             0.0
