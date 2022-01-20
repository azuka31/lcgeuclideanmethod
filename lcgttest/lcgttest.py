# LCG T-TEST USING EUCLIDEAN METHOD
# ---------------------------------
# Advanced Analytics and Growth Marketing Telkomsel
# -------------------------------------------------
# Project Supervisor : Rizli Anshari
# Writer             : Azka Rohbiya Ramadani, Muhammad Gilang, Demi Lazuardi

from os import name
from matplotlib.markers import MarkerStyle
from numpy.core.fromnumeric import size
import pandas as pd
import math
import numpy as np
from pandas.io.pytables import incompatibility_doc
import matplotlib.pyplot as plt
from scipy.stats import t, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

plt.style.use('seaborn')

class EuclideanMethod():
    """Determining LCG and Random Takers by calculating optimum p-value"""
    def __init__(self, df_takers, df_nontakers, chunk = 3, ascending = True):
        
        # data preparation
        self.df_takers = EuclideanMethod.load_df(df_takers)
        self.df_nontakers = EuclideanMethod.load_df(df_nontakers)

        # number of each chunk
        self.chunksize = EuclideanMethod.takers_chunk(self, chunk)

        # general info
        self.rev_takers_min = self.df_takers.rev.min()
        self.rev_takers_max = self.df_takers.rev.max()
        self.rev_takers_sd = self.df_takers.rev.std(ddof = 0)
        self.rev_takers_avg = self.df_takers.rev.mean()
        self.subs = len(self.df_takers)
        
        # temporary table for chunk takers
        self.df_takers_chunk = EuclideanMethod.df_takers_chunk(self)
        
        #avg chunk --> to determine ED
        self.takers_chunk_avg = EuclideanMethod.takers_chunk_avg(self)

        # temporary table non_takers with ED
        self.df_nontakers_ED = EuclideanMethod.generate_ed(self, ascending)

    def load_df(df):
        """To create df with standard naming and by default sorted by rev

        Parameters
        ----------
        filename :
            name of file in csv
        df :
            dataframe with two column, msisdn and delta_rev (rev)

        Returns
        -------
        type
            dataframe

        """
        column_name = ['msisdn','rev']
        df.set_axis(column_name, axis = 1, inplace = True)
        df.sort_values(by='rev', inplace = True)
        df.reset_index(inplace = True, drop = True)
        return df
    
    def takers_chunk(self, chunk):
        """To devide how many rows for each chunk to create segment

        Parameters
        ----------
        chunk :
            3 by default

        Returns
        -------
        type
            list of chunk size

        """
        total_row_takers = len(self.df_takers.index)
        num_segment = total_row_takers//chunk
        remaining_segment = total_row_takers%chunk

        main_segment = np.array([num_segment for i in range(chunk)])
        add_segment = []
        for i in range(chunk):
            if i < remaining_segment:
                add_segment.append(1)
            else:
                add_segment.append(0)
        add_segment = np.array(add_segment)

        segment = main_segment + add_segment
        return segment
    
    def df_takers_chunk(self):
        """Creating segment of total population """
        chunksize = self.chunksize

        x = 0
        df = []
        for i in chunksize:
            idx = x
            fdx = x + (i-1)
            df.append(self.df_takers.loc[idx:fdx])
            x += i
        return df
    
    def takers_chunk_avg(self):
        """Calculating average each chunksize """
        takers_chunk_avg = []
        for df in self.df_takers_chunk:
            takers_chunk_avg.append(df.rev.mean())
        return takers_chunk_avg

    def generate_ed(self, ascending = True):
        """Adding new column of Euclidean Distance as the table preparation before determining t-test value """

        df = self.df_nontakers.copy()
        avg_chunk = self.takers_chunk_avg

        ED_name = ['ED{}'.format(num+1) for num in range(len(self.chunksize))]

        for num in range(len(self.chunksize)):
            name_col = ED_name[num]
            avg = avg_chunk[num]
            df[name_col] = df.rev.apply(lambda x : abs(x-avg))
            # print(df[name_col])

        df['min_ED'] = df[ED_name].min(axis = 1)

        df_columns = df.columns
        df.sort_values(by = 'min_ED', ascending = ascending, inplace = True)
        df.reset_index(inplace = True)
        return df[df_columns]
    
    def info(self):
        """General info """
        
        info = [
        self.subs,
        self.rev_takers_avg,
        self.rev_takers_sd,
        self.rev_takers_min,
        self.rev_takers_max,
        self.chunksize,
        self.takers_chunk_avg]
        label = [
        'subs',
        'rev_takers_avg',
        'rev_takers_sd',
        'rev_takers_min',
        'rev_takers_max',
        'chunksize',
        'takers_chunk_avg']
        return pd.DataFrame({'label' : label, 'info' : info})

    def processes_final_table(self, m, idx = 0, end_idx = None, comment1 = ''):
        """
        Main process to calculating final table of p-value

        Parameters
        ----------
        m :
            calculation step to determine p-value
        idx :
             (Default value = 0)
        end_idx :
             (Default value = None)

        Returns
        -------
            Dataframe containing p-value (as main column) and calculation of ttest. Population with the maximum of p-value is on our purpose
        """
        # spliting target table
        if end_idx != None:
            df = self.df_nontakers_ED[['msisdn','rev']]
            df = df.loc[:(end_idx-1)]
        else:
            df = self.df_nontakers_ED[['msisdn','rev']]

        total_row = len(df.index)
        total_row_progress = len(df.index)- idx

        # Validation of m value
        if m >= total_row:
            raise ValueError('Multification Value must be under total_row')

        total_loop = 1 + (total_row_progress//m) if total_row_progress%m != 0 else (total_row_progress//m)
        
        df_result = []
        df_columns = ['label','population','avg_revenue','diff_avg','sd','diff_sd','t_value','t_df','p_value','min_','max_']

        job = 0
        step = idx
        while step < total_row:
            # print('Launching {a} of {b}'.format(a=job,b=total_loop))
            EuclideanMethod.progressbar((job+1), total_loop, comment1=comment1)
            # print('job ',job)
            # print('total',total_loop)
            job += 1
            step += m
            if (total_row - step) > 0:
                idx += m
            else:
                idx = total_row
            
            if idx != 1:
                label = '1 - {}'.format(idx)
                avg = df.loc[:(idx-1)].rev.mean()
                rev_avg = avg/self.rev_takers_avg - 1
                std = df.loc[:(idx-1)].rev.std(ddof = 0)
                rev_std = std/self.rev_takers_sd - 1
                min_ = df.loc[:(idx-1)].rev.min()
                max_ = df.loc[:(idx-1)].rev.max()

                t_value = abs((avg - self.rev_takers_avg)/math.sqrt((std**2/idx)+(self.rev_takers_sd**2/self.subs)))
                t_df = ((((std**2)/(idx)) + ((self.rev_takers_sd**2)/self.subs))**2) / ((((std**2/idx)**2)/(idx-1)) + ((((self.rev_takers_sd**2)/self.subs)**2)/(self.subs-1)))
                p_value = t.sf(abs(t_value), df = t_df)*2
                row = [label, idx, avg, rev_avg, std, rev_std, t_value, t_df, p_value, min_, max_]
                df_result.append(row)

        print()
        df_result = pd.DataFrame(df_result)
        result = df_result.set_axis(df_columns, axis = 1)

        max_p_value = result.p_value.max()
        optimum_lcg = result.population[result.p_value == max_p_value].min()
        

        return result
     
    def run(self, m_custom = 1, comment = True, comment1 = ""):
        """
        Process main, with step 1 as main process

        Parameters
        ----------
        m_custom :
             (Default value = 1)

        Returns
        -------
            Dataframe containing p-value (as main column) and calculation of ttest. Population with the maximum of p-value is on our purpose

        """
        if comment:
            text = '''\nRESULT HAS BEEN RUN SUCCESSFULLY\n--------------------------------\n\nOutput\n------\n\n    main\n    ----\n    df_result\n    tukey\n    df_tukey\n    summary\n\n    plot\n    ----\n    plot_p_value()\n    plot_tukey()\n\n'''
        else:
            text = ''
        if m_custom < 1:
            raise ValueError("Step must be more than 0")
        else:
            self.df_result = EuclideanMethod.processes_final_table(self, m = m_custom, comment1= comment1)
            summary = EuclideanMethod.summarize(self)
            self.summary = summary.copy()
            self.df_tukey = EuclideanMethod.combine_table(self.df_nontakers_ED, self.df_takers, self.optimum_population)
            self.tukey = EuclideanMethod.tukey_table_generator(self)
            print(text)
            return self.df_result

    def run_parsial(self, m, idx, end_idx, comment = True):
        """
        Process main, with step 1 as main process

        Parameters
        ----------
        m :
           Step of iteration to determine p-value 
        idx :
            Start index for slicing table.
        end_idx :
            End index for slicing table

        Returns
        -------
            Dataframe containing p-value (as main column) and calculation of ttest. Population with the maximum of p-value is on our purpose

        """
        if comment:
            text = '''\n\nRESULT HAS BEEN RUN SUCCESSFULLY\n--------------------------------\n\nOutput\n------\n\n    main\n    ----\n    df_result_parsial\n    df_result\n    tukey\n    df_tukey\n    summary\n\n    plot\n    ----\n    plot_p_value()\n    plot_tukey()\n'''
        else:
            text = ''
        if (m < 1) or (idx < 0) or (end_idx > len(self.df_nontakers)):
            raise ValueError('Arguments are out of ranges')
        else:
            df = EuclideanMethod.processes_final_table(self, m, idx, end_idx)
            self.df_result_parsial = df.copy()
            result = pd.concat([self.df_result,df]).drop_duplicates().sort_values(by='population').reset_index(drop=True)
            self.df_result = result.copy()
            summary = EuclideanMethod.summarize(self)
            self.summary = summary.copy()
            self.df_tukey = EuclideanMethod.combine_table(self.df_nontakers_ED, self.df_takers, self.optimum_population)
            self.tukey = EuclideanMethod.tukey_table_generator(self)
            print(text)
            return result
    
    def help_output(self):
            text = '''\nRESULT HAS BEEN RUN SUCCESSFULLY\n--------------------------------\n\nOutput\n------\n\n    main\n    ----\n    df_result\n    tukey\n    df_tukey\n    summary\n\n    plot\n    ----\n    plot_p_value()\n    plot_tukey()\n\n'''
            print(text)

    def tukey_table_generator(self):
        """ """

        tukey = pairwise_tukeyhsd(endog = self.df_tukey['rev'],
                                    groups = self.df_tukey['category'],
                                    alpha = 0.05)

        tukey_data = tukey.summary().data[1:]
        tukey_columns = tukey.summary().data[0]
        df = pd.DataFrame(tukey_data, columns = tukey_columns)
        
        return df
        # return tukey
    
    def combine_table (non_takers_ED,takers,n_index): 
        """
        Concat table containing non_takers, takers, and random

        Parameters
        ----------
        non_takers_ED :
            From self.non_takers
            
        takers :
            From self.takers
            
        n_index :
            Optimum index for determininng LCG, population with highest p-value
            

        Returns
        -------
            Dataframe of Random, LCG, and and Takers

        """
        df1 = pd.DataFrame(non_takers_ED)
        df1 = df1[['msisdn','rev']]
        df1['category'] = pd.Series(["Random" for x in range(len(df1.index))])
        
        df2 = pd.DataFrame(takers)
        df2 = df2[['msisdn','rev']]
        df2['category'] = pd.Series(["Takers" for x in range(len(df2.index))])
        
        #input value with the highest pvalue (close to 1), result from file_table_output.csv
        df3 = df1.copy()
        df3 = df3[:n_index]
        df3['category'] = pd.Series(["LCG_Alike" for x in range(len(df3.index))])
        #df3.to_csv ('file_table_output_x.csv', index = False, header=True)
        
        frames = [df1, df2, df3]
        result_df = pd.concat(frames, ignore_index=True)
        
        return result_df
            
    def summarize(self):

        df = self.df_result
        max_p_value = df.p_value.max()
        optimum_average = np.array(list(map(abs,df.diff_avg)))

        # posible bug here, statisical error
        optimum_lcg = df.population[df.p_value == max_p_value].min()
        self.optimum_population = optimum_lcg
        self.max_p_value = max_p_value

        df_summary = df[(df.p_value == max_p_value) & (df.population == optimum_lcg)].T
        df_summary.set_axis(['value'], axis=1, inplace=True)
        return df_summary

    def plot_p_value(self,col = 'tab:blue', label = None):
        """
        Plot p value based on population
        """
        try:
            EuclideanMethod._plot_p_value(self, col, label)
            plt.title('P Value vs Population')

            plt.xlabel('population')
            plt.ylabel('p value')
            
            plt.legend()
            plt.show()
        except:
            raise ValueError('run method must be executed first')

    def _plot_p_value(self, col = 'tab:blue', label = None):
        """
        Plot p value based on population
        """
        try:
            column = ['population','p_value']
            df = self.df_result[column]
            plt.plot(df.population, df.p_value, color = col, label = label)
            plt.plot(df.population, df.p_value, alpha = 0, label ='Max Value: {:.4f}\n Population : {}'.format(self.max_p_value, self.optimum_population))
        except:
            raise ValueError('run method must be executed first')
    
    def plot_tukey(self):
        """ 
        Plot tukey to illustrate Multiple Comparison of Means
        """
        EuclideanMethod._plot_tukey(self)
        plt.legend()
        plt.yticks(self._plot_range, self._tab['name'])
        plt.title("Multiple Comparison of Means - Tukey HSD, FWER = 5%", loc='left')
        plt.tight_layout()
        plt.show()

    def _plot_tukey(self):
        """ """
        try: 
            df = self.tukey.copy() 
            df['name'] = df['group1']+' - '+df['group2']
            tab = df[['name','lower','meandiff','upper']]

            plot_range = range(1,len(tab.index)+1)
            plt.hlines(y=plot_range, xmin=tab['lower'], xmax=tab['upper'], color='grey', alpha=0.3)
            plt.scatter(tab['lower'], plot_range, color='skyblue', alpha=1, label='lower',s = 100)
            plt.scatter(tab['upper'], plot_range, color='green', alpha=0.8 , label='upper', s= 100)
            plt.scatter(tab['meandiff'], plot_range, color='tab:red', alpha=0.3 , label='meandiff', marker = '|')
            # plt.axvline(0, color = 'red', linewidth = 0.3, alpha = 0.3)
            self._plot_range = plot_range
            self._tab = tab
        except:
            raise ValueError('run method must be executed first')

    def plot_sd(self):
        """ """
        x = self.df_result.row_right_table.values
        y = self.df_result.diff_sd.values

        plt.style.use('ggplot')

        plt.plot(x,y)
        plt.xlabel('row_total')
        plt.ylabel('diff_sd')
        plt.show()
        # pass

    def plot_dist(self):
        """ """
        return
    
    def progressbar(x, total, comment1):
        """
        Parameters
        ----------
        x :
            Percentage of progress 
        total :
            Total Progress

        Returns
        -------
            Printing realtime progress
        """
        size = 30
        progress = x / total
        multi = math.ceil(progress*size)
        arrow = '#'*multi + ' '*(size-multi)
        print("Computing {} : {}| {:2.1%} running jobs ".format(comment1, arrow, progress), end="\r")

class MapEuclideanMethod(EuclideanMethod):
    """ Same as Euclidean Method but using map for calculating muliple value"""
    def __init__(self, arr_df_takers, arr_df_nontakers, label = None, chunk = 3, ascending = True, comment = True, commentMethod=""):
        if label == None:
            label = ['pack{}'.format((i+1)) for i in range(len(arr_df_takers))]

        self.arr_df_takers = arr_df_takers
        self.arr_df_nontakers = arr_df_nontakers
        self.chunk = [chunk for i in range(len(self.arr_df_takers))]
        self.ascending = [ascending for i in range(len(self.arr_df_takers))]
        self.label = label

        MapEuclideanMethod.validate(self)

        self.result = list()
        self.classes = map(EuclideanMethod, self.arr_df_takers, self.arr_df_nontakers, self.chunk, self.ascending)
        # print(MapEuclideanMethod.infomap(self))
        dict_nontakes_ED = {}
        dict_info = {}
        dict_df_result = {}
        dict_tukey = {}
        dict_df_tukey = {}
        dict_summary = {}

        
        
        for idx,cls in enumerate(self.classes):

            df_ed = cls.df_nontakers_ED   
            dict_nontakes_ED[label[idx]] = df_ed.copy() 

            info = cls.info()
            info.set_axis(['variable_takers','{}_info'.format(label[idx])], axis=1, inplace=True)
            dict_info[label[idx]] = info
            print(info)
            
            cls.run(comment=False, comment1='{}{}/{}'.format(commentMethod,(idx+1),len(self.label)))
            dict_df_result[label[idx]] = cls.df_result.copy()
            dict_tukey[label[idx]] = cls.tukey
            dict_df_tukey[label[idx]] = cls.df_tukey

            summary = cls.summary
            if idx == 0:
                dict_summary['index'] = summary.index
            dict_summary[label[idx]] = summary.value.copy()

        self.dict_nontakers_ED = dict_nontakes_ED   
        self.dict_df_result = dict_df_result
        self.dict_tukey = dict_tukey
        self.dict_df_tukey = dict_df_tukey
        self.dict_info = dict_info
        summary = pd.DataFrame(dict_summary)
        self.df_summary = summary.set_index('index')
        
        if comment == True:
            print('''\n\nRESULT HAS BEEN RUN SUCCESSFULLY\n--------------------------------\n\nOutput\n------\n    prep\n    ----\n    dict_nontakers_ED\n    dict_info\n\n    main\n    ----\n    dict_df_result\n    dict_tukey\n    df_summary\n\n    plot\n    ----\n    plot_all_p_value()\n    plot_all_tukey()\n\n        ''')

    def validate(self):
        """Validating that 2 arrays have same length """
        if len(self.arr_df_takers) != len(self.arr_df_nontakers):
            raise ValueError('List must be in the same length')
        else:
            pass
    
    def help_output(self):
        print('''\n\nRESULT HAS BEEN RUN SUCCESSFULLY\n--------------------------------\n\nOutput\n------\n    prep\n    ----\n    dict_nontakers_ED\n    dict_info\n\n    main\n    ----\n    dict_df_result\n    dict_df_tukey\n    dict_tukey\n    df_summary\n\n    plot\n    ----\n    plot_all_p_value()\n    plot_all_tukey()\n\n        ''')

    def plot_all_p_value(self, sort = False):
        """ Plotting p value of multiple object"""
     
        n_rows, n_columns = MapEuclideanMethod.plot_grid_rules(self)
        i = 1 
        for package, df in self.dict_df_result.items():
            plt.subplot(n_rows, n_columns, i)
            plt.plot(df['population'],df['p_value'])
            if sort == False:
                plt.xlabel(package)
            else:
                plt.xlabel('{} ({})'.format(package, self.sort_label[(i-1)]))
            i+=1
        plt.suptitle('P-Value Chart')
        # plt.tight_layout()
        plt.show()

    def plot_all_tukey(self, sort = False):
        """ Plotting tukey of multiple object"""
        n_rows, n_columns = MapEuclideanMethod.plot_grid_rules(self)
        i = 1
        for package, df in self.dict_tukey.items():
            plt.subplot(n_rows, n_columns, i)
            self.tukey = df
            EuclideanMethod._plot_tukey(self)
            
            if sort == False:
                plt.xlabel(package)
            else:
                plt.xlabel('{} ({})'.format(package, self.sort_label[(i-1)]))
    
            if (i-1)%n_columns == 0:
                plt.yticks(self._plot_range, self._tab['name'])
            else:
                plt.yticks([])
            i+=1

        plt.legend(bbox_to_anchor=(0.5,-1.2), ncol = 3)
        plt.suptitle("Tukey HSD, FWER = 5%")
        # plt.tight_layout()
        plt.show()
    
    def plot_grid_rules(self):
        """ Grid rules to show plot multiple object"""
        len_df = len(self.dict_tukey)
        n_rows = math.ceil(len_df/4)
        if len_df >= 4:
            n_columns = 4
        else:
            n_columns = len_df
        return n_rows, n_columns

class EuclideanMethodAscDesc():
    '''Running MapEuclidean Method by either ascending and descending'''
    def __init__(self, arr_df_takers, arr_df_nontakers, label = None, chunk = 3):
        self.asc = MapEuclideanMethod(arr_df_takers, arr_df_nontakers, label = label, chunk = chunk, ascending = True, comment = False, commentMethod='Ascending Method ')
        self.desc = MapEuclideanMethod(arr_df_takers, arr_df_nontakers, label = label, chunk =chunk,  ascending = False, comment = False, commentMethod='Descending Method ')

        EuclideanMethodAscDesc.compare_average(self)
        EuclideanMethodAscDesc.filter_result(self)

        EuclideanMethodAscDesc.help_output(self)

    def compare_average(self):
        '''Comparing average value between ascending and descending method'''
        label = []
        asc_takers_avg = []
        asc_nontakers_avg = [] 
        desc_nontakers_avg = [] 
        for package, info in self.asc.dict_info.items():
            label.append(package)
            asc_takers_avg.append(info.loc[1].values[1])
            asc_nontakers_avg.append(self.asc.df_summary[package].loc['avg_revenue'])
            desc_nontakers_avg.append(self.desc.df_summary[package].loc['avg_revenue'])
        
        df = pd.DataFrame({'package' : label,
                             'takers_avg' : asc_takers_avg,
                             'asc_nontakers_avg' : asc_nontakers_avg,
                             'desc_nontakers_avg': desc_nontakers_avg, 
                             'asc_delta_avg' : np.array(asc_takers_avg) - np.array(asc_nontakers_avg),
                             'desc_delta_avg' : np.array(asc_takers_avg) - np.array(desc_nontakers_avg)}
                            )
        temp_x = abs(df.asc_delta_avg) <= abs(df.desc_delta_avg)
        df['opt_sort'] = temp_x.apply(lambda x : 'asc' if x == True else 'desc')
        self.df_asc_desc_avg = df.copy() 
        df_corr = df.corr()
        asc_pearson = df_corr.loc['takers_avg']['asc_nontakers_avg']
        desc_pearson = df_corr.loc['takers_avg']['desc_nontakers_avg']
        self.asc_person = asc_pearson
        self.desc_person = desc_pearson
    
    def help_output(self):
        
        print('''\nASCENDING DESCENDING METHODS HAVE BEEN APPLIED\n----------------------------------------------\n\n    Preparation:\n        asc\n        desc\n\n    output:\n        df_asc_desc_avg\n        dict_df_result\n        dict_df_tukey\n        dict_tukey\n        df_summary\n\n    visualization:\n        plot_compare_hist()\n        plot_avg_ascdesc()\n        plot_compare_p_value()\n        plot_all_p_value()\n        plot_all_tukey()''')
        
    def filter_result(self):
        '''Deciding wheter ascending or descending method for each pacakge'''
        summary = pd.DataFrame()
        dict_df_tukey = {}
        dict_tukey = {}
        dict_df_result = {}

        main_df = self.df_asc_desc_avg.copy()
        col_targ = ['package','opt_sort']
        df = main_df[col_targ]

        for package, opt_sort in zip(df.package, df.opt_sort):
            if opt_sort == 'asc':
                tmp_series = self.asc.df_summary[package].copy()
                dict_df_result[package] = self.asc.dict_df_result[package].copy()
                dict_df_tukey[package] = self.asc.dict_df_tukey[package].copy()
                dict_tukey[package] = self.asc.dict_tukey[package]
            else:
                tmp_series = self.desc.df_summary[package].copy()
                dict_df_result[package] = self.desc.dict_df_result[package].copy()
                dict_df_tukey[package] = self.desc.dict_df_tukey[package].copy()
                dict_tukey[package] = self.desc.dict_tukey[package]

            summary[package] = tmp_series.append(pd.Series({'opt_sort_method':opt_sort}))
        
        self.df_summary = summary.copy()
        self.dict_df_result = dict_df_result
        self.dict_df_tukey = dict_df_tukey
        self.dict_tukey = dict_tukey
        self.sort_label = df.opt_sort.values
    
    def plot_all_p_value(self):
        '''Plot all best (asc/dsc) pacakge'''
        MapEuclideanMethod.plot_all_p_value(self, sort = True)

    def plot_all_tukey(self):
        '''Plot all best (asc/dsc) pacakge'''
        MapEuclideanMethod.plot_all_tukey(self)
        
    def plot_avg_ascdesc(self):
        '''Plot comparison of average asc and desc'''
        
        df = self.df_asc_desc_avg.copy()
        plt.plot(df['takers_avg'], df['asc_nontakers_avg'], marker = '*', label = 'asc_nontakers, pearson {}'.format(self.asc_person))
        plt.plot(df['takers_avg'], df['desc_nontakers_avg'], marker = 'o', label = 'desc_nontakers, pearson {}'.format(self.desc_person))
        plt.xlabel('AVG Takers')
        plt.ylabel('AVG Non Takers (LCG)')
        plt.legend()
        plt.show()

    def plot_compare_hist(self):
        '''histogram of takers and nontakers for each package'''

        self.dict_tukey = self.asc.dict_tukey
        n_rows, n_columns = MapEuclideanMethod.plot_grid_rules(self)
        i = 1 
        for idx, df in enumerate(self.asc.arr_df_takers):
            df2 = self.asc.arr_df_nontakers[idx]
            plt.subplot(n_rows, n_columns, i)
            plt.hist(df.rev, alpha = 0.5, label = 't')
            plt.hist(df2.rev, alpha = 0.5, label = 'n')
            plt.xlabel(self.asc.label[idx])
            i+=1
        plt.suptitle('Histogram')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_compare_p_value(self):
        """ Plotting comparison p value ascending and descending"""
        self.dict_tukey = self.asc.dict_tukey
        n_rows, n_columns = MapEuclideanMethod.plot_grid_rules(self)
        i = 1 
        for package, df in self.asc.dict_df_result.items():
            df2 = self.desc.dict_df_result[package]
            plt.subplot(n_rows, n_columns, i)
            plt.plot(df['population'],df['p_value'])
            plt.plot(df2['population'],df2['p_value'])
            plt.xlabel(package)
            i+=1
        plt.suptitle('P-Value Chart')
        plt.tight_layout()
        plt.show()

# file preprocessing
def dfpack2arr(file_takers_perpack, file_nontakers_perpack, from_pandas = False):
    """
    Parameters
    ----------
    file_takers_perpack :
        File takers containing msisdn, package type, rev before, rev after, and delta 
    file_nontakers_perpack :
        File nontakers containing msisdn, package type, rev before, rev after, and delta 

    Returns
    -------
        df_takers, df_nontakers, labels (in list type)
    """
    
    takers = dfpack2dict(file_takers_perpack, from_pandas=from_pandas)
    nontakers = dfpack2dict(file_nontakers_perpack, from_pandas=from_pandas)

    if takers['label'] != nontakers['label']:
        raise ValueError('Pack label need to be check to make sure that both csv files are related')
    else:
        pass 

    df_takers = takers['data']
    df_nontakers = nontakers['data']
    labels = takers['label']

    return df_takers, df_nontakers, labels

def dfpack2dict(filename, from_pandas = False):

    if from_pandas == False:
        df = pd.read_csv(filename)
    else:
        df = filename.copy()

    standard_name = ['msisdn','pack','rev_before','rev_after','rev']
    df.set_axis(standard_name, axis=1, inplace=True)
    df.sort_values(by='rev', inplace=True, ignore_index=True)
    
    df['pack'] = [ pack.lower() for pack in df.pack]
    df['pack'] = [ pack.replace(" ",'') for pack in df.pack]
    
    unique_pack = sorted(df.pack.unique())
    arr_df = []

    for pack in unique_pack:
        target = ['msisdn','rev_before']
        df_n = df[target][df.pack == pack]
        df_n.reset_index(inplace=True)
        arr_df.append(df_n[target])
    return {'label': unique_pack, 'data':arr_df}
    