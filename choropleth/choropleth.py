#! /usr/bin/env python3
import os, sys, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from matplotlib.collections import LineCollection
from matplotlib import cm
import pandas as pd
import shapefile
import mpl_toolkits.basemap
import shapefile


def getopt(clf, ret_val, isbool=False):
    """ Command Line Option input parser"""
    found = []
    def getCLO(flag):
        iindx = sys.argv.index(flag)
        sys.argv.pop(iindx)
        return sys.argv.pop(iindx)
    if isbool: return (clf in sys.argv)
    while clf in sys.argv: found.append(getCLO(clf))
    if found: ret_val = [found, found[0]][int(len(found) == 1)]
    return ret_val

def scrubComments(filename):
    """Helper function for JSON loading"""
    comment_chars=['//','#']
    def not_startswith(string): return not bool(sum([ int(string.strip().startswith(xx)) for xx in comment_chars]))
    with open(filename) as ff: return ''.join([ll for ll in filter(lambda x: not_startswith(x), ff.readlines()) ])

def main():
    o_dict={'datafile':getopt('-f','./data/trauma_cost_study_20140604.xlsx'),
            'config_file':  getopt('-c','./config/config.json'),
            'output':  getopt('-o','./output/choropleth_data.xlsx')}
    ch=ChoroHandler(**o_dict)
    ch()
    plt.show()
def onpick(event):
  print( "Zip Code: {}\n".format(event.artist.get_label()) )
class DataFile:
    def __getattr__(self, name): return self.__dict__.get(name, None)
    def __init__(self, *args, **kwargs):
        self._handle_args(*args)
        self._handle_kwargs(**kwargs)
    def _handle_args(self, *args):pass
    def _handle_kwargs(self, **kwargs):
        self.__dict__.update(kwargs)
    def _parse(self, df=None):
        if self.filename:
            if os.path.splitext(self.filename)[1]=='.xlsx': df=pd.read_excel(self.filename, 'Summary')
            elif os.path.splitext(self.filename)[1]=='.csv': df=pd.read_csv(self.filename)
            else: print('Error parsing "{}"'.format(self.filename))
        return df
    @property
    def df(self):
        if not self._df: self._df=self._parse()
        return self._df
    @df.setter
    def df(self, value): self._df=value

class ChoroBase(object):
    def __init__(self, *args, **kwargs):
        self._handle_args(*args)
        self._handle_kwargs(**kwargs)
    def __getattr__(self, name): return self.__dict__.get(name, None)
    def _handle_args(self, *args): pass
    def _handle_kwargs(self, **kwargs): self.__dict__.update(kwargs)


class Choropleth(ChoroBase):
    def Denver(self):
        #return [-105.25, -104.55, 39.5, 40.0]
        return [-110.6373, -99.3671, 36.8058, 43.2809]
    def Colorado(self):
        return [-109.05, -102.05,37.0,41.0 ]
    def _prep_fig(self):
        countDF=pd.DataFrame({'Count':self.df[self.data]})
        self.countDF=countDF.sort('Count')#, ascending=False)
        # zipIndx=self.countDF['Count'].index

        cntMax,cntMin=float(self.countDF['Count'].max()),float(self.countDF['Count'].min())
        fig = plt.figure(figsize=(11.7,8.3))
        cmap=plt.cm.jet
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=cntMin, vmax=cntMax))
        sm._A = []
        plt.subplots_adjust(left=0.05,right=0.95,top=0.90,bottom=0.05,wspace=0.15,hspace=0.05)
        ax = plt.subplot(111)
        ax.set_title(self.title)
        return fig, sm, ax

    def __call__(self, *args, **kwargs):
        if self.df is None or self.data is None: return
        fig, sm, ax=self._prep_fig()
        m=self._prep_basemap()
        [ax.add_collection(line) for line in self._prep_shapefile(self.zcta_shapefile,m)]
        [ax.add_collection(line) for line in self._prep_shapefile(self.road_shapefile,m)]
        cbar = plt.colorbar(sm,format='%1i')
        #cbar.ax.set_yticklabels([int(cntMin),int(cntMin+((cntMax-cntMin)/3)),int(cntMin+(2*(cntMax-cntMin)/3)), int(cntMax)])
        # if 'Cost' in title:
        #     cbar.set_label('x $100,000.00')#, rotation=270)
        fig.canvas.mpl_connect('pick_event', onpick)
        self.fig=fig
        #plt.ylim([-110.6373, -99.3671])
        #plt.xlim([39.5, 40.0])
        #plt.xlim([588466,685487])
        #plt.ylim([369600,466621])
        if self.region=='denaur':
            plt.xlim([588466,685487])
            plt.ylim([369600,466621])
        if self.region=='denmet':
            plt.xlim([528466,785487])
            plt.ylim([319600,516621])
        if self.region=='co':
            plt.xlim([130832,979827])
            plt.ylim([16949,615999])

        # COMMENTED OUT FOR SPEED
        # print(fig)
        # plt.show()
        if self.save_it: plt.savefig('%s.png'%(title.replace(' ','_')+'_'+self.region))


    def _prep_basemap(self):
        [x1, x2, y1, y2] = self.Denver()
        m = Basemap(resolution='i',projection='merc', llcrnrlat=y1,urcrnrlat=y2,llcrnrlon=x1,urcrnrlon=x2)#,lat_ts=90)
        m.drawcoastlines(linewidth=0.2)
        m.drawcountries(linewidth=0.2)
        m.drawstates(linewidth=0.2)

        #Reference: http://lance-modis.eosdis.nasa.gov/imagery/subsets/?subset=AERONET_BSRN_BAO_Boulder.2013351.aqua.250m
        im=plt.imread('./data/AERONET_BSRN_BAO_Boulder.2013351.aqua.250m.tif')
        m.imshow(im, extent=self.Colorado())
        return m

    def _prep_shapefile(self, shapefile_dir,m, road=False):
        r = shapefile.Reader(shapefile_dir)#"./data/305113/tigerline_shapefile_2010_2010_state_colorado_2010_census_5-digit_zip_code_tabulation_area_zcta5_state-based")
        roadList=['I-70','I-225','I-25','USHWY36','COLFAX','E-470','USHWY285','I-76','USHWY6','HWYE-470','HWY470']
        cmap=plt.cm.jet

        def handle_road(record, shape):
            lons,lats = zip(*shape.points)
            data = np.array(m(lons, lats)).T
            if sum([int(road in record[2].upper().replace(' ','')) for road in roadList]) > 0:
                if len(shape.parts) == 1: segs = [data,]
                else: segs=[data[shape.parts[ii-1]:shape.parts[ii]] for ii in range(1,len(shape.parts)) ]+[data[len(shape.parts):]]
                lines = LineCollection(segs,antialiaseds=(1,))
                lines.set_edgecolors('w')
                lines.set_linewidth(2)
                return lines
        def handle_zip(record, shape):
            cntMax,cntMin=float(self.countDF['Count'].max()),float(self.countDF['Count'].min())
            lons,lats = zip(*shape.points)
            data = np.array(m(lons, lats)).T
            if len(shape.parts) == 1: segs = [data,]
            else: segs=[data[shape.parts[ii-1]:shape.parts[ii]] for ii in range(1,len(shape.parts)) ]+[data[len(shape.parts):]]
            if record[0] in self.countDF['Count'].index:
                pop=float(self.countDF['Count'][record[0]])
                lines = LineCollection(segs,antialiaseds=(1,))
                color=cmap((pop-cntMin)/( 0.00000000001 if (cntMax-cntMin) == 0 else cntMax-cntMin) )[:3]
                lines.set_facecolors(color)
                lines.set_label(record[0])
                lines.set_picker(True)
            else:
                lines = LineCollection(segs,antialiaseds=(1,))
            lines.set_edgecolors('k')
            lines.set_linewidth(0.1)
            return lines

        func=handle_road if road else handle_zip
        line_list=[func(record, shape) for record, shape in zip(r.records(),r.shapes())]
        return [ ll for ll in line_list if ll is not None]


class ChoroHandler(ChoroBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._handle_kwargs(**kwargs)
    def __call__(self, *args, **kwargs):
        if self.datafile and os.path.isfile(self.datafile):
            self.df=DataFile(filename=self.datafile).df
            self.strip_data()
            self.upper_data()
            self.apply_user_defined()
            zc_sf="./data/305113/tigerline_shapefile_2010_2010_state_colorado_2010_census_5-digit_zip_code_tabulation_area_zcta5_state-based"
            rd_sf="./data/tl_2010_08_prisecroads/tl_2010_08_prisecroads"
            self.cp=Choropleth(df=self.extracted_df, data='HHI', title='Household Income',zcta_shapefile=zc_sf,road_shapefile=rd_sf)
            self.cp()
            # plt.show()
    def apply_user_defined(self):
        if self.config and self.config.get('Distill',None):
            for kk,vv in self.config.get('Distill').items():
                self.apply_distill(col_name=kk, **vv)
        if self.config and self.config.get('Extract',None):
            self.extracted_df=self.extract(self.config.get('Extract'))
            print(self.extracted_df)

    def _extract(self, group_by=None, action=None, column=None, filter_by=None):
        if group_by and action and column and not filter_by:
            return self.df.groupby(group_by)[column].__getattribute__(action)()
        if group_by and action and column and filter_by:
            return self.df[self.df[filter_by]==True].groupby(group_by)[column].__getattribute__(action)()

    def extract(self, extract_dict=None):
        if extract_dict:
            return pd.DataFrame({kk:self._extract(**vv) for kk, vv in extract_dict.items()})

    def apply_distill(self, col_name=None, column=None, query=None, type='icontains'):
        if type == 'icontains':
            if col_name and column and query:
                if isinstance(query, list): self.df[col_name]=self.df.apply(lambda row: ( bool(sum( [qq.upper() in row[column].upper() for qq in query] ) > 0) if isinstance(row[column],str) else False), axis=1)
                else: self.df[col_name]=self.df.apply(lambda row: (query.upper() in row[column].upper() if isinstance(row[column],str) else False), axis=1)
        if type == 'contains':
            if col_name and column and query:
                if isinstance(query, list): self.df[col_name]=self.df.apply(lambda row: ( bool(sum( [qq in row[column] for qq in query] ) > 0) if isinstance(row[column],str) else False), axis=1)
                else: self.df[col_name]=self.df.apply(lambda row: (query in row[column] if isinstance(row[column],str) else False), axis=1)

    def strip_data(self): [self.df.__setattr__(column, self.df[column].map(lambda x: (x.strip() if isinstance(x, str) else x))) for column in self.df.columns]
    def upper_data(self): [self.df.__setattr__(column, self.df[column].map(lambda x: (x.upper() if isinstance(x, str) else x))) for column in self.df.columns]
    def _handle_kwargs(self, **kwargs):
        self.__dict__.update(kwargs)
        if self.config_file:
            self.config=json.loads(scrubComments(self.config_file))


if __name__=="__main__":
    main()
