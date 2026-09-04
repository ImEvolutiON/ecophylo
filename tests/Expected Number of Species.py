import ecophylo
import numpy as np

n = 20000 # How many individuals are sampled
mu_distrib = [1e-7, 1e-3, "log_unif"] # distribution law for speciation rate
No_distrib = [1e7, 1e12, "log_unif"] # distribution law for No
mu = ecophylo.sample(lower = mu_distrib[0], upper = mu_distrib[1], distr = mu_distrib[2]) # draw a value
No = ecophylo.sample(lower = No_distrib[0], upper = No_distrib[1], distr = No_distrib[2]) # draw a value

mu = 1e-5
No = 2e6
deme_sizes = [[No]]
seed = 1
after_merge_species_list = []
before_merge_species_list = []
before_merge_singleton_list = []
after_merge_singleton_list = []
before_merge_tmrca = []
after_merge_tmrca = []
before_merge_abund_list = []
after_merge_abund_list = []
getAbund_species_list = []
getAbund_total_abundances_list = []

end = 101

for i in range(1, end):
    tree, debug_info = ecophylo.simulate(
    samples = n,
    deme_sizes = deme_sizes,
    mu = mu,
    verbose = False,
    spmodel = 'paraphyletic',
    age = "nearest",
    tau = 0,
    seed = seed,
    debug = True)
    
    print(f'Simulation {i}/{end-1} done')
    
    seed += 1
    before_merge_species_list.append(debug_info["before_merge"]["n_species"])
    after_merge_species_list.append(debug_info["after_merge"]["n_species"])
    before_merge_abund_list.append(debug_info["before_merge"]["total_abundance"])
    after_merge_abund_list.append(debug_info["after_merge"]["total_abundance"])
    before_merge_singleton_list.append(debug_info["before_merge"]["n_singletons"])
    after_merge_singleton_list.append(debug_info["after_merge"]["n_singletons"])
    before_merge_tmrca.append(debug_info["before_merge"]["tmrca"])
    after_merge_tmrca.append(debug_info["after_merge"]["tmrca"])
    # test with getAbund() function to test reliability between total_abundances
    abund = ecophylo.getAbund(tree, spmodel = 'lacy')
    getAbund_species_list.append(len(abund))
    getAbund_total_abundances_list.append(sum(abund))

print("\n\n========= ABUNDANCES CHECK ============")
print(f'Expected Number of Individuals (our fixed n) : {n}')
print(f'Mean Number of Individuals over {end-1} simulations before the merge : {np.mean(before_merge_abund_list)}')
print(f'Mean Number of Individuals over {end-1} simulations after the merge : {np.mean(after_merge_abund_list)}')
print(f'Mean Number of Individuals over {end-1} simulations with getAbund() : {np.mean(getAbund_total_abundances_list)}')





print("\n\n========= RICHNESS CHECK ============")
theta = 2*No*mu
i = np.arange(n)+1
ewen = np.sum(theta / (theta + i - 1)) # Tavaré 2021 "Magic Formula" says E(Kn) = sum(theta/(theta+i-1))
print(f'Expected Number of Species (Ewen\'s Sampling) : {ewen}')
ewen_simplification = theta*np.log(1+(n/theta))
print(f'Expected Number of Species (Ewen\'s Sampling Simplification for n<<No) : {ewen_simplification}')
print(f'Mean Number of species for {end-1} simulations before merging process : {np.mean(before_merge_species_list)}')
print(f'Mean Number of species for {end-1} simulations after merging process : {np.mean(after_merge_species_list)}')
print(f'Mean Number of species for {end-1} simulations with getAbund() : {np.mean(getAbund_species_list)}')




print("\n\n========= SINGLETON CHECK ============")
Expected_singleton = (theta*n)/(n-1+theta) # Yun S. Song 2021 "Lecture Notes on computational and Mathematical Population Genetics" says Proposition 5.7
                              # (Excpected number of uniquely represented haplotypes). Let Hn,1 denote the number of distinct haplotypes represented
                              # exactly once in a sample of size n. Then, E(Hn,1) = n*theta/(n-1+theta)
print(f'Expected Number of singletons : {Expected_singleton:.10f}')
print(f'Mean Number of singletons for {end-1} simulations before merge : {np.mean(before_merge_singleton_list)}')
print(f'Mean Number of singletons for {end-1} simulations after merge : {np.mean(after_merge_singleton_list)}')



print("\n\n========= TMRCA CHECK ============")

E_TMRCA = 2*No*(1-(1/n)) # Yun S. Song 2021 says Theorem 1.10 (Expected TMRCA in coalescent units under constant population size)
                         # E(Wn) = E_TMRCA = 2(1-(1/n)) in coalescent unit, 1 coalescent unit is 2Ne so E(Wn) = 2Ne * 2(1-(1/n))
                         # However, we are using haploids individuals : we need to divide by 2.
print(f'Expected TMRCA : {60*E_TMRCA}')
print(f'Mean TMRCA for 100 simulations before merge : {60*np.mean(before_merge_tmrca)}')
print(f'Mean TMRCA for 100 simulations after merge : {60*np.mean(after_merge_tmrca)}')


















