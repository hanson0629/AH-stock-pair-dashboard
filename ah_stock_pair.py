import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
import akshare as ak
from tqdm import tqdm
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

start_date = '20240101'
end_date = datetime.now().strftime('%Y-%m-%d').replace('-', '') # 获取到当前日期
ex_rate = 0.937  # 港币兑人民币汇率

def get_dual_listed_stocks():
    # 获取A股列表
    stock_a = ak.stock_info_a_code_name()
    stock_a = stock_a[['code', 'name']]
    
    # 获取H股列表 - 使用最新接口
    try:
        stock_h = ak.stock_hk_spot()
        stock_h = stock_h[['代码', '名称']].rename(columns={'代码': 'code', '名称': 'name'})
    except:
        try:
            stock_h = ak.stock_hk_spot_em()
            stock_h = stock_h[['代码', '名称']].rename(columns={'代码': 'code', '名称': 'name'})
        except Exception as e:
            print(f"获取港股列表失败: {str(e)}")
            return pd.DataFrame()
    
    # 找出同名公司(简化匹配逻辑)
    dual_listed = pd.merge(stock_a, stock_h, on='name', how='inner', suffixes=('_A', '_H'))
    
    # 手动添加一些知名的双重上市公司
    known_dual = [
        {'name': '中信证券', 'code_A': '600030', 'code_H': '06030'},
        {'name': '中国平安', 'code_A': '601318', 'code_H': '02318'},
        {'name': '招商银行', 'code_A': '600036', 'code_H': '03968'},
        {'name': '中国铝业', 'code_A': '601600', 'code_H': '02600'},
    ]
    
    return pd.concat([dual_listed, pd.DataFrame(known_dual)], ignore_index=True).drop_duplicates()

def get_stock_history(code, market):
    try:
        if market == 'A':
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date,end_date=end_date, adjust="")
            df = df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume'
            })
        else:
            try:
                df = ak.stock_hk_hist(symbol=code, period="daily", start_date=start_date,end_date=end_date, adjust="")
            except:
                print(f"获取港股数据失败: {str(e)}")
        
        df = df.rename(columns={
                    '日期': 'date',
                    '开盘': 'open',
                    '收盘': 'close',
                    '最高': 'high',
                    '最低': 'low'
                })
        return df.set_index('date')[['open', 'high', 'low', 'close']].dropna()
    except Exception as e:
        print(f"获取{code}数据失败: {str(e)}")
        return None

def preprocess_data(a_df, h_df, a_code):
    if a_df is None or h_df is None or len(a_df) < 30 or len(h_df) < 30:
        return None
    
    # 合并数据，只保留共同交易日
    merged = pd.merge(
        a_df[['close']], 
        h_df[['close']], 
        left_index=True, 
        right_index=True,
        how='inner', 
        suffixes=('_A', '_H')
    )
    
    # 检测并剔除A股涨跌停日
    a_df['prev_close'] = a_df['close'].shift(1)
    a_df['pct_change'] = (a_df['close'] - a_df['prev_close']) / a_df['prev_close']
    
    is_st = 'ST' in a_code or '*' in a_code
    limit_threshold = 0.049 if is_st else 0.099
    
    limit_up_dates = a_df[a_df['pct_change'] >= limit_threshold].index
    limit_down_dates = a_df[a_df['pct_change'] <= -limit_threshold].index
    
    merged = merged[
        ~merged.index.isin(limit_up_dates) & 
        ~merged.index.isin(limit_down_dates)
    ]
    
    # 添加汇率调整(港币换算为人民币)
    merged['close_H'] = merged['close_H'] * ex_rate
    
    return merged.dropna()

def enhanced_cointegration_analysis(a_prices, h_prices, name, a_code=None, h_code=None):
    try:
        # 关键变更点：交换y和x的位置
        # 现在 y = A股价格, x = 港股价格
        coint_stat, coint_pvalue, _ = coint(a_prices, h_prices, autolag='AIC')
        
        # 协整回归 y = α + βx + ε
        X = sm.add_constant(h_prices)  # 自变量是港股价格
        model = sm.OLS(a_prices, X).fit()  # 因变量是A股价格
        const, beta = model.params
        resid = model.resid
        
        # ADF检验
        adf_result = adfuller(resid, autolag='AIC')
        
        # 计算价差
        spread = a_prices - (const + beta * h_prices)
        
        # 获取最新价格
        latest_a_price = a_prices.iloc[-1] if not a_prices.empty else None
        latest_h_price = h_prices.iloc[-1] if not h_prices.empty else None
        
        # 结果整理
        result = {
            '名称': name,
            'A股代码': a_code,
            '港股代码': h_code,
            'A股价格': latest_a_price,
            '港股价格(人民币)': latest_h_price,
            '协整关系': f"A = {const:.4f} + {beta:.4f}×H",
            'α值': const,
            'β值': beta,
            #'协整统计量': coint_stat,
            #'协整_p值': coint_pvalue,
            #'残差均值': resid.mean(),
            '残差标准差': resid.std(),
            '残差z-score': (resid.iloc[-1] - resid.mean()) / resid.std() if not resid.empty else None,
            #'残差ADF统计量': adf_result[0],
            '残差ADF_p值': adf_result[1],
            '样本数量': len(a_prices),
            #'A股平均价': a_prices.mean(),
            #'港股平均价(人民币)': h_prices.mean(),
            #'价差比率': (a_prices.mean() - h_prices.mean()) / h_prices.mean() if h_prices.mean() != 0 else None,
            #'价差均值': spread.mean(),
            #'价差标准差': spread.std()
        }
        
        return result, resid, spread
        
    except Exception as e:
        st.error(f"{name} 协整分析失败: {str(e)}")
        return None, None, None

def run_analysis():
    st.text("获取双重上市公司列表...")
    dual_listed = get_dual_listed_stocks()
    st.text(f"找到{len(dual_listed)}家双重上市公司")
    
    results = []
    residuals = {}
    spreads = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, row in enumerate(dual_listed.iterrows()):
        progress = idx / len(dual_listed)
        progress_bar.progress(progress)
        
        _, row = row
        a_code = row['code_A']
        h_code = row['code_H']
        name = row['name']
        
        status_text.text(f"分析中: {name} (A股:{a_code}, 港股:{h_code})")
        
        a_df = get_stock_history(a_code, 'A')
        h_df = get_stock_history(h_code, 'H')
        merged = preprocess_data(a_df, h_df, a_code)
        
        if merged is None or len(merged) < 100:
            continue
            
        analysis, resid, spread = enhanced_cointegration_analysis(
            a_prices=merged['close_A'],
            h_prices=merged['close_H'],
            name=name,
            a_code=a_code,
            h_code=h_code
        )
        
        if analysis:
            analysis.update({
                'A股代码': a_code,
                '港股代码': h_code
            })
            results.append(analysis)
            residuals[name] = resid
            spreads[name] = spread
    
    progress_bar.progress(1.0)
    status_text.text("分析完成!")
    
    return results, residuals, spreads

def main():
    st.set_page_config(page_title="A-H股票对分析仪表盘", page_icon=":chart_with_upwards_trend:", layout="wide")
    
    st.title("A-H股票对分析仪表盘")
    st.markdown("""该仪表盘展示了A股和港股双重上市公司的协整分析结果，帮助识别潜在的套利机会。""")
    
    # 侧边栏配置
    st.sidebar.header("分析参数")
    
    # 添加日期选择器
    global start_date
    start_date_input = st.sidebar.date_input(
        "开始日期",
        datetime.strptime(start_date, '%Y%m%d'),
        min_value=datetime(2010, 1, 1),
        max_value=datetime.now()
    )
    start_date = start_date_input.strftime('%Y%m%d')
    
    # 添加汇率调整
    global ex_rate
    ex_rate = st.sidebar.number_input("港币兑人民币汇率", value=ex_rate, step=0.001, format="%.3f")
    
    # 添加运行按钮
    if st.sidebar.button("运行分析"):
        results, residuals, spreads = run_analysis()
        
        if results:
            result_df = pd.DataFrame(results)
            
            # 确保协整_p值列存在
            if '协整_p值' in result_df.columns:
                result_df = result_df.sort_values('协整_p值')
                significant = result_df[result_df['协整_p值'] < 0.05]
            else:
                significant = result_df
            
            # 显示结果表格
            st.header("分析结果")
            
            # 添加过滤选项
            st.subheader("所有股票对")
            
            # 创建可排序的交互式表格
            st.dataframe(result_df, use_container_width=True)
            
            if '协整_p值' in result_df.columns and not significant.empty:
                st.subheader("显著协整的股票对 (p值 < 0.05)")
                st.dataframe(significant, use_container_width=True)
            
            # 添加详细分析部分
            st.header("详细分析")
            
            # 选择股票进行详细分析
            selected_stock = st.selectbox("选择股票查看详细分析", list(residuals.keys()))
            
            if selected_stock:
                st.subheader(f"{selected_stock}的详细分析")
                
                # 获取选中股票的数据
                selected_data = result_df[result_df['名称'] == selected_stock].iloc[0]
                
                # 显示基本信息
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("A股代码", selected_data['A股代码'])
                    if 'A股价格' in selected_data:
                        st.metric("A股价格", f"{selected_data['A股价格']:.2f}")
                
                with col2:
                    st.metric("港股代码", selected_data['港股代码'])
                    if '港股价格(人民币)' in selected_data:
                        st.metric("港股价格(人民币)", f"{selected_data['港股价格(人民币)']:.2f}")
                
                with col3:
                    if '协整关系' in selected_data:
                        st.metric("协整关系", selected_data['协整关系'])
                    if '残差z-score' in selected_data:
                        st.metric("当前残差的z-score", f"{selected_data['残差z-score']:.2f}")
                
                # 绘制残差时间序列图
                if selected_stock in residuals and residuals[selected_stock] is not None:
                    st.subheader("残差时间序列")
                    resid_series = residuals[selected_stock]
                    
                    # 计算移动平均和标准差带
                    resid_df = pd.DataFrame({
                        '残差': resid_series,
                        '均值': resid_series.rolling(window=20).mean(),
                        '上限 (+2σ)': resid_series.rolling(window=20).mean() + 2 * resid_series.rolling(window=20).std(),
                        '下限 (-2σ)': resid_series.rolling(window=20).mean() - 2 * resid_series.rolling(window=20).std()
                    })
                    
                    fig = px.line(resid_df, y=['残差', '均值', '上限 (+2σ)', '下限 (-2σ)'])
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 添加残差分布直方图
                    st.subheader("残差分布")
                    fig = px.histogram(resid_series, nbins=50, marginal="box")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("没有找到符合条件的股票对")
    else:
        st.info("请点击侧边栏中的'运行分析'按钮开始分析。")

if __name__ == '__main__':
    main()
