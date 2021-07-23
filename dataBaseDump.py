import mysql.connector
import pandas as pd


class DataBaseDump:

    def create_data_dump(self):
        cnx = mysql.connector.connect(
            user='root', password='1234', database='chatbotdb2', host='127.0.0.1'
        )
        cursor = cnx.cursor()

        query = (
            "select  usr.gender gender,cm.chat_member chat_member ,cm.chat_message chat_message, im.category item_category,im.discount_percentage item_discount,ci.quantity order_quantity, im.price item_price,  od.order_amount order_total_amount,od.state_of_order order_status,usr.age from cart_item ci inner join chat_message cm on ci.ip_address =  cm.ip_address inner join item im on ci.item_id = im.item_id inner join user usr on usr.user_id=ci.user_id inner join order_details od  on od.order_id= ci.order_detail_id "
        )

        cursor.execute(query)
        res = cursor.fetchall()
        csv_doc = pd.DataFrame(res, columns=['gender', 'chat_member', 'chat_message', 'item_category', 'item_discount',
                                             'order_quantity', 'item_price', 'order_total_amount', 'order_status',
                                             'age'])
        csv_doc.to_csv('chat1.csv')
        cursor.close()
        cnx.close()

    def create_data_dump_ip_address(self):
        cnx = mysql.connector.connect(
            user='root', password='1234', database='chatbotdb2', host='127.0.0.1'
        )
        cursor = cnx.cursor()

        query = (
            "select  cm.chat_member chat_member ,cm.chat_message chat_message, im.category item_category,im.discount_percentage item_discount,ci.quantity order_quantity, im.price item_price,  od.order_amount order_total_amount,od.state_of_order order_status from cart_item ci inner join chat_message cm on ci.ip_address =  cm.ip_address inner join item im on ci.item_id = im.item_id inner join order_details od  on od.order_id= ci.order_detail_id "
        )

        cursor.execute(query)
        res = cursor.fetchall()
        csv_doc = pd.DataFrame(res, columns=['chat_member', 'chat_message', 'item_category', 'item_discount',
                                             'order_quantity', 'item_price', 'order_total_amount', 'order_status'])
        csv_doc.to_csv('chat2.csv')
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
