import numpy as np
sqrt=np.sqrt; pi=np.pi; conj=np.conjugate; LA=np.linalg
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import os, pickle, time
import defns, projections as proj, analysis_defns as AD, group_theory_defns as GT
from F3 import F3_mat



def main():
  # Choose what scenario to test

  #irreps = GT.irrep_list() 
  irreps = ['A1+']

  shell = 'all'
  l = 'both'

  E_MIN = 4.2
  E_MAX = 4.3
  L = 5

  show_eigs = False

  find_root = True
  E0 = 4.25 # initial guess for root-finder
  order = 1 # polynomial order used for fit

  new_plots = False
  (Ymin,Ymax) = (-1e6,1e6)
  (ymin,ymax) = (-1e5,1e5)


  # Separate energies by # of active shells
  breaks = defns.shell_breaks(E_MAX,L)
  breaks.append(E_MAX)
  for b in range(1,len(breaks)-1): # skip first window
    Emin = breaks[b]+1e-8
    Emax = breaks[b+1]-1e-8 if breaks[b+1]!=E_MAX else breaks[b+1]
    if Emax<E_MIN or Emin>E_MAX:
      continue
    else:
      Emin = max(Emin,E_MIN)
      Emax = min(Emax,E_MAX)    

    #########################################################
    # Define parameters (necessary for several functions)

    # K2 parameters
    
    # a0=-10; r0=0.5; P0=0.5; a2=-1
    # K2_dir = 'a0=m10_r0=0.5_P0=0.5_a2=m1/'

    #a0=0.1; r0=0; P0=0; a2=0
    
    #a0=0.1; r0=0; P0=0; a2=0.1

    #a0=0.1; r0=0; P0=0; a2=0.3

    #a0=0.1; r0=0; P0=0; a2=0.5

    #a0=0.1; r0=0; P0=0; a2=0.7

    #a0=0.1; r0=0; P0=0; a2=0.9

    a0=0.1; r0=0; P0=0; a2=1


    #a0=0; r0=0; P0=0; a2=0.1
    
    #a0=0; r0=0; P0=0; a2=0.3

    #a0=0; r0=0; P0=0; a2=0.5

    #a0=0; r0=0; P0=0; a2=0.7
    
    #a0=0; r0=0; P0=0; a2=0.9
    
    #a0=0; r0=0; P0=0; a2=1


    # F2_KSS parameter
    alpha=0.5  # so far I've ALWAYS set alpha=0.5

    
    # Data & plot directories
    K2_dir = 'a0='+str(a0)+'_r0='+str(r0)+'_P0='+str(P0)+'_a2='+str(a2)+'/'
    data_dir = 'Data/'+K2_dir
    plot_dir = 'Plots/'+K2_dir
    
    if not os.path.exists(data_dir):
      os.makedirs(data_dir)
    if not os.path.exists(plot_dir):
      os.makedirs(plot_dir)
  

    # Total CM energy & lattice size (lists)
    E_list = [2.9,2.95,#2.99,2.995,2.999,2.9999,
    3.0001,#3.001,3.005,3.01,
    #3.0318609166490167, # A1+ single root
    3.03186092,
    3.05,3.1,3.15,
    #3.16303178,
    #3.1630317882, # T1+ triple root
    3.16303179,
    3.2,3.25,3.3,
    #3.30679060158175, # T2- triple root
    3.30679061,
    3.35,3.4,3.45,3.5,3.55,3.6,
    3.61,3.62,3.63,3.65,3.67,3.68,3.69,3.7,3.75,3.8,3.81,3.82,3.827,
    3.82755621, # just below where (1,1,0) turns on
    #
    #################################
    # Begin 3-shell regime
    3.84,3.85,3.86,3.87,3.88,3.89,
    #3.8950494797495034, # T2+ triple root
    3.89504948,
    3.9,3.91,3.92,3.93,
    #3.9360802243295736, # E- double root
    3.93608023,
    3.94,
    #3.9485642787554895, # A2- single root
    3.94856428,
    3.95,3.96,
    #3.963,3.965,3.967,3.96798888,
    #3.9679888998546713, # old root of Ftilde before removing q's
    #3.96798890,3.968,3.969,
    3.97,#3.973,3.975,3.977,
    3.98,3.99,
    4.0,4.05,4.1,
    #4.105402464984292, # T2- triple root
    4.10540247,
    4.15,4.16,4.17,4.19,4.195,4.2,4.205,
    #4.209892475540663, # T1+ triple root
    4.20989248,
    4.21,
    4.21193816,
    4.21193817,
    #4.211938171368993, # non-interacting energy E1         #############
    4.21193818,
    #4.21193947,
    4.21193948, # A1+ single root for a0=0, a2=0.1 ?
    #4.21193949,
    4.2125,4.213,4.2135,4.214,4.2145,4.215,
    4.22,4.23,4.24,
    #4.2421632582173645, # E+ double root for a0=0.1, a2=0 (r0=P0=0)
    4.24216326,
    4.25,4.27,
    #4.2784624639738, # A1+ single root for a0=0.1, a2=0 (r0=P0=0)
    4.27846247,
    4.3,4.32,4.33,#4.33374164,
    #4.333741640225551, # A1+ single root
    4.33374165,
    4.335,4.338,4.34,
    #4.341716880828459, # T1- triple root
    4.34171689,
    4.342,4.345,
    4.35,4.37,4.4,4.43,
    #4.441146045889443, # T2- triple root
    4.44114605,
    4.45,
    #4.486997310056035, # T1+ triple root
    4.48699732,
    4.5,4.55,
    4.58101788 # just below where (1,1,1) turns on
    ]
    
    E_list = [x for x in E_list if Emin<=x<=Emax]
    L_list = [5.0]

    # print('E='+str(E)+', L='+str(L)+'\n')
    # print('Shells: ', defns.shell_list(E,L), '\n')
    # print('Matrix dimension: ', len(defns.list_nnk(E,L))*6, '\n')

    ####################################################################################
    # Load F3 matrix from file if exists, otherwise compute from scratch & save to file

    for L in L_list:
      F3_list=[]
      
      for E in E_list:
        if a2==0:
          datafile = data_dir+'F3_00_E'+str(E)+'_L'+str(L)+'.dat'
        elif a0==0:
          datafile = data_dir+'F3_22_E'+str(E)+'_L'+str(L)+'.dat'
        else:
          datafile = data_dir+'F3_E'+str(E)+'_L'+str(L)+'.dat'

        try:
          with open(datafile, 'rb') as fp:
            F3 = pickle.load(fp)
            #F3 = pickle.loads(fp.read())
            print('F3 loaded from '+datafile+'\n')

        except IOError:
          print(datafile+' not found; computing F3 from scratch...')
          t0 = time.time() 
          if a2==0:
            F3 = F3_mat.F3mat00(E,L,a0,r0,P0,a2,alpha)
          elif a0==0:
            F3 = F3_mat.F3mat22(E,L,a0,r0,P0,a2,alpha)
          else:
            F3 = F3_mat.F3mat(E,L,a0,r0,P0,a2,alpha)
          t1=time.time()
          print('Calculation complete (time:',t1-t0,')')
          with open(datafile, 'wb') as fp:
            pickle.dump(F3, fp, protocol=4, fix_imports=False)
            print('F3 saved to '+datafile+'\n')

        F3_list.append( F3 )

      ###################################################
      # Create list of inputs needed by several functions
      inputs = [L,a0,r0,P0,a2,alpha]

      # Free energies
      E_free = defns.E_free_list(L,3,1).values()
      E_free = [e for e in E_free if Emin<e<Emax]


      ##################################################
      # Project onto chosen irrep
      for flag in irreps:

        # General irrep
        if flag in GT.irrep_list():
          I = flag
          inputs.append(I)

          if a2==0:
            if sum([GT.subspace_dim_o_l(s,I,0) for s in defns.shell_list(Emax,L)]) == 0:
              print("0-dim l'=l=0 subspace for "+I+' for E<'+str(round(Emax,4)))
              continue

            irrep_eigs_array_list = AD.F3i_00_I_eigs_list(E_list,L,F3_list,I)  

            irrep_eigs_array_list_flip = AD.F3i_00_I_eigs_list(E_list,L,F3_list,I,flip=True)

            f_eigs = F3_mat.F3i_00_I_eigs

          elif a0==0:
            if sum([GT.subspace_dim_o_l(s,I,2) for s in defns.shell_list(Emax,L)]) == 0:
              print("0-dim l'=l=2 subspace for "+I+' for E<'+str(round(Emax,4)))
              continue
            
            irrep_eigs_array_list = AD.F3i_22_I_eigs_list(E_list,L,F3_list,I)  

            irrep_eigs_array_list_flip = AD.F3i_22_I_eigs_list(E_list,L,F3_list,I,flip=True)

            f_eigs = F3_mat.F3i_22_I_eigs
            
          else:
            if sum([GT.subspace_dim_o(s,I) for s in defns.shell_list(Emax,L)]) == 0:
              print('0-dim subspace for '+I+' for E<'+str(round(Emax,4)))
              continue

            irrep_eigs_array_list = AD.F3i_I_eigs_list(E_list,L,F3_list,I)  

            irrep_eigs_array_list_flip = AD.F3i_I_eigs_list(E_list,L,F3_list,I,flip=True)

            f_eigs = F3_mat.F3i_I_eigs


          if show_eigs==True:
            for i in range(len(E_list)):
              E = E_list[i]
              x = irrep_eigs_array_list[i]
              #x = [y for y in x if abs(y)<1e3]
              print(E,x)
              #print(E,min(x,key=abs))

          if find_root==True:
            root = AD.root_finder_secant(E_list,irrep_eigs_array_list,f_eigs,inputs,E0,order)
            print(root)

            # irrep_roots_list = AD.root_finder(E_list, irrep_eigs_array_list, f_eigs, inputs)

          
          

          #print(irrep_eigs_array_list)
          
          #print(irrep_roots_list)
          #print(root)

          #irrep_roots_file = data_dir+I+'_roots_L='+str(L)+'.dat'
          #with open(irrep_roots_file,'w') as fp:
          #  fp.write(str(irrep_roots_list))

        
        #################################################
        # # BAD isotropic approx (A1+ proj, l=0)
        # elif flag == 'iso':
        #   iso_eigs_array_list = AD.F3i_iso_eigs_list_bad(E_list,L,F3_list)
          
        #   iso_roots_list = AD.root_finder(E_list, iso_eigs_array_list, F3_mat.F3i_iso_eigs_bad, inputs)
        #   # print(iso_roots_list)

        #   iso_roots_file = data_dir+'iso_roots_L='+str(L)+'.dat'
        #   with open(iso_roots_file,'w') as fp:
        #     fp.write(str(iso_roots_list))


        # #################################################
        # # A1+ projection (l=0 and l=2 contributions)
        # elif flag == 'A1':
        #   A1_eigs_array_list = AD.F3i_A1_eigs_list(E_list,L,F3_list)
        
        #   A1_roots_list = AD.root_finder(E_list, A1_eigs_array_list, F3_mat.F3i_A1_eigs, inputs)
          
        #   #print(A1_roots_list)

        #   A1_roots_file = data_dir+'A1_roots_L='+str(L)+'.dat'
        #   with open(A1_roots_file,'w') as fp:
        #     fp.write(str(A1_roots_list))


        #################################################
        # Full matrix, no projections (doesn't work well)
        elif flag=='full':
          eigs_array_list = AD.F3i_eigs_list(E_list,L,F3_list)

          roots_list = AD.root_finder(E_list, eigs_array_list, F3_mat.F3i_eigs, inputs)
          # print(roots_list)

          roots_file = data_dir+'roots_L='+str(L)+'.dat'
          with open(roots_file,'w') as fp:
            fp.write(str(roots_list))

        

        #################################################
        # Plot F3i eigenvalues vs. E
        if new_plots == True:
          plotfile1 = plot_dir+str(flag)+'_'+str(b+1)+'shells.pdf'
          plotfile2 = plot_dir+str(flag)+'_'+str(b+1)+'shells_zoom.pdf'

          if len(flag)==2:
            irrep_tex = '$'+flag[0]+'^'+flag[1]+'$'
          elif len(flag)==3 and flag!='iso':
            irrep_tex = '$'+flag[0]+'_'+flag[1]+'^'+flag[2]+'$'

          plt.plot(E_list,irrep_eigs_array_list,'.')
          plt.plot(E_list,irrep_eigs_array_list_flip,'o',mfc='none')
          plt.xlabel('E'); plt.ylabel(r'$\lambda$')
          plt.title(r'$F_3^{-1}$ eigenvalues, '+irrep_tex+' irrep, '+str(b+1)+' shells')

          for e0 in E_free:
            plt.axvline(x=e0,c='k',ls='--',lw=1)
          
          plt.xlim((Emin,Emax))
          plt.ylim((Ymin,Ymax))
          plt.tight_layout(); plt.grid(True)
          plt.savefig(plotfile1)
          plt.savefig('tmp.pdf')

          plt.ylim((ymin,ymax))
          plt.savefig(plotfile2)
          plt.savefig('tmp2.pdf')

          plt.close()



          # Plot only the eigenvalues closest to 0

          # small_eigs = AD.small_eig_list(eigs_array_list)
          # plt.figure()
          # plt.plot(E_list,small_eigs,'.')
          # plt.grid(True)
          # plt.savefig('tmp2.pdf')


if __name__=='__main__':
  main()
