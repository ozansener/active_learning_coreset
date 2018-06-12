import numpy as np

def fisher_information(alld,already_selected,remaining,delta,probs, budget):
  C = probs.shape[1]
  info = np.zeros((probs.shape[0]))
  for c in range(C):
    pi=probs[:,c]
    one_minus_pi = 1 - pi
    inf_c = np.multiply(pi,one_minus_pi)
    info = info + inf_c
  # Info is the I vector
  xxtranspose = np.matmul(alld,alld.transpose())
  new_ones = []
  info[already_selected] = 0
  chosen_x = xxtranspose[already_selected,:]
  denom = np.sum(np.multiply(np.multiply(chosen_x,chosen_x),info), axis=0)+delta
  best_f = 0
  i = 0
  while len(new_ones)<budget:
    i = i + 1
    best_f = 0
    for candidate in remaining:
      cand_remember = info[candidate]
      info[candidate] = 0
      inc = np.multiply(xxtranspose[candidate,:],xxtranspose[candidate,:])
      inc = np.multiply(inc, info)
      denom_cand = inc + denom
      new_f = (1.0/delta)*(np.sum(info)+cand_remember) - np.sum(np.divide(info,denom_cand))
      if new_f > best_f:
        best_f = new_f
        best_cand = candidate
      info[candidate] = cand_remember
    print "Choose {} as {} of {}".format(best_cand,i,budget)
    already_selected = np.append(already_selected, [best_cand])
    remaining_l = list(remaining)
    remaining_l.remove(best_cand)
    remaining = np.array(remaining_l)
    new_ones.append(best_cand)
    info[best_cand] = 0
    inc = np.multiply(xxtranspose[best_cand,:],xxtranspose[best_cand,:])
    inc = np.multiply(inc, info)
    denom = inc + denom
  return new_ones
