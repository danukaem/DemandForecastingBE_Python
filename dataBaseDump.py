import mysql.connector
import pandas as pd


class DataBaseDump:

    def create_model_1_csv_file(self):
        cnx = mysql.connector.connect(
            user='root', password='1234', database='chatbotdb2', host='127.0.0.1'
        )
        cursor = cnx.cursor()

        query = (
            "select age,district,gender, occupation,item_code from `user` ur inner JOIN order_details od on ur.user_id = od.user_id INNER JOIN cart_item ci on od.order_id = ci.order_detail_id INNER JOIN item im on ci.item_id = im.item_id where od.state_of_order =0"
        )

        cursor.execute(query)
        res = cursor.fetchall()
        csv_doc = pd.DataFrame(res, columns=['age', 'district', 'gender', 'occupation', 'item_code'])
        csv_doc.to_csv('model_1.csv')
        cursor.close()
        cnx.close()

    def create_model_2_csv_file(self):
        cnx = mysql.connector.connect(
            user='root', password='1234', database='chatbotdb2', host='127.0.0.1'
        )
        cursor = cnx.cursor()

        query = (
            "select age,gender,occupation,district,brand,color,item_category,ram,price,screen,processor,itm_ordr.item_code from item_extract_rasa ier INNER JOIN `user` ur on ur.user_id = ier.user_id LEFT JOIN order_details od on ier.session_id = od.session_id  LEFT JOIN (select ord.order_id order_id ,ord.session_id  session_id,ord.state_of_order state_of_order ,im.item_id item_id , im.item_code from order_details ord INNER JOIN cart_item ci on  ord.order_id = ci.order_detail_id INNER JOIN item im on ci.item_id = im.item_id) itm_ordr on itm_ordr.session_id = ier.session_id where itm_ordr.state_of_order = 0  "
        )

        cursor.execute(query)
        res = cursor.fetchall()
        csv_doc = pd.DataFrame(res,
                               columns=['age', 'gender', 'occupation', 'district', 'brand', 'color', 'item_category',
                                        'ram', 'price', 'screen', 'processor', 'item_code'])
        csv_doc.to_csv('model_2.csv')
        cursor.close()
        cnx.close()

    def get_query_data(self, sql_query):
        cnx = mysql.connector.connect(
            user='root', password='1234', database='chatbotdb2', host='127.0.0.1'
        )
        cursor = cnx.cursor()

        query = (
            sql_query
        )

        cursor.execute(query)
        res = cursor.fetchall()
        cursor.close()
        cnx.close()
        return res
