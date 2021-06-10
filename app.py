import flask
from flask import render_template
import pickle
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return render_template('main.html' )
        
    if flask.request.method == 'POST':
        temp = 0
        answer = 'Клиент останется'
        with open('ensemble.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
        state = (flask.request.form['state'])
        acc_len = float(flask.request.form['account_length'])
        inter_plan = int(flask.request.form['international_plan'])
        v_mail_plan = int(flask.request.form['voice_mail_plan'])
        number_v_msgs = float(flask.request.form['number_vmail_messages'])
        t_day_min = float(flask.request.form['total_day_minutes'])
        t_day_call = float(flask.request.form['total_day_calls'])
        t_day_char = float(flask.request.form['total_day_charge'])
        t_eve_min = float(flask.request.form['total_eve_minutes'])
        t_eve_call = float(flask.request.form['total_eve_calls'])
        t_eve_char = float(flask.request.form['total_eve_charge'])
        t_night_min = float(flask.request.form['total_night_minutes'])
        t_night_call = float(flask.request.form['total_night_calls'])
        t_night_char = float(flask.request.form['total_night_charge'])
        t_intl_min = float(flask.request.form['total_intl_minutes'])
        t_intl_call = float(flask.request.form['total_intl_calls'])
        t_intl_char = float(flask.request.form['total_intl_charge'])
        custom_serv_call = float(flask.request.form['customer_service_calls'])

        df = pd.DataFrame(data=[[state, acc_len, inter_plan, v_mail_plan,
                                 number_v_msgs, t_day_min, t_day_call, t_day_char,
                                 t_eve_min, t_eve_call, t_eve_char, t_night_min,
                                 t_night_call, t_night_char, t_intl_min,
                                 t_intl_call, t_intl_char, custom_serv_call]],
                          columns=['State', 'Account length', 'International plan',
                          'Voice mail plan', 'Number vmail messages', 'Total day minutes',
                          'Total day calls', 'Total day charge', 'Total eve minutes',
                          'Total eve calls', 'Total eve charge', 'Total night minutes',
                          'Total night calls', 'Total night charge', 'Total intl minutes',
                          'Total intl calls', 'Total intl charge', 'Customer service calls'])

        state_groups = {'AK': 0, 'AL': 1, 'AR': 3, 'AZ': 0, 'CA': 3, 'CO': 1,
                        'CT': 2, 'DC': 0, 'DE': 2, 'FL': 1, 'GA': 2, 'HI': 0,
                        'IA': 0, 'ID': 1, 'IL': 0, 'IN': 1, 'KS': 3, 'KY': 1,
                        'LA': 0, 'MA': 2, 'MD': 3, 'ME': 3, 'MI': 3, 'MN': 2,
                        'MO': 1, 'MS': 3, 'MT': 3, 'NC': 2, 'ND': 1, 'NE': 0,
                        'NH': 2, 'NJ': 3, 'NM': 1, 'NV': 3, 'NY': 2, 'OH': 1,
                        'OK': 2, 'OR': 2, 'PA': 2, 'RI': 0, 'SC': 3, 'SD': 1,
                        'TN': 0, 'TX': 3, 'UT': 2, 'VA': 0, 'VT': 1, 'WA': 3,
                        'WI': 0, 'WV': 0, 'WY': 1}
        df['State'] = df['State'].map(state_groups)

        scaler = StandardScaler()
        X_std = scaler.fit_transform(df)

        if 'send' in flask.request.form:
            temp = loaded_model.predict(df)
            if temp == 1:
                answer = 'Клиент уйдет'
            else:
                answer = 'Клиент останется'
        return render_template('main.html', result = answer)

if __name__ == '__main__':
    app.run()
