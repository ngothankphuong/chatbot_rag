from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk import Action
from rasa_sdk.events import SlotSet
from actions import Constant
constant  = Constant()

from custom.write_file import write_file
import logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')



class actionSuMangTamNhinTrietLyGiaoDuc(Action):
    def name(self):
        return "action_thong_tin_truong"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            info_truong = constant.get_thong_tin_truong()	

            entities = tracker.latest_message.get('entities', [])
            print(f"\nEntities từ input:\n {entities}")
            
            default_response = "Xin lỗi tôi không có thông tin về câu hỏi này! Vui lòng cung cấp chi tiết thông tin."
            responses = ""

            for entity in entities:
                if entity.get('entity') == 'thong_tin_truong':
                    role_truong = entity.get('role')
                    
                    print(f"{role_truong}")
                    
                    if role_truong:
                        part_response = info_truong.get(role_truong,)
                        if part_response:
                            responses += part_response
                        else:
                            file_writer = write_file()
                            file_writer.get_ghi_log_file('action_thong_tin_dang: ' + tracker.latest_message['text'])
                            
                    else:
                        file_writer = write_file()
                        file_writer.get_ghi_log_file('action_thong_tin_dang: ' + tracker.latest_message['text'])
                    
            
            if responses:
                dispatcher.utter_message(text=responses)
            else:
                dispatcher.utter_message(text=default_response)
        except Exception as e:
            file_writer = write_file()
            file_writer.get_ghi_log_file(f"Exception in action_thong_tin_truong: {str(e)}")
            dispatcher.utter_message(text=default_response)
        return []


class ActionBanGiamHieu(Action):
    def name(self):
        return "action_ban_giam_hieu"
   
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
        # Tạo dictionary ánh xạ các role với thông điệp tương ứng
            info_ban_giam_hieu = constant.get_info_ban_giam_hieu()
                
            entities = tracker.latest_message.get('entities', [])
            print(f"\nEntities từ input:\n {entities}")
            default_response = "Xin lỗi tôi không có thông tin về câu hỏi này! Vui lòng cung cấp chi tiết thông tin."
            responses = ""
           
            for entity in entities:
                if entity.get('entity') == 'ban_giam_hieu':
                    role_ban_giam_hieu = entity.get('role')
                   
                    print(f"{role_ban_giam_hieu}")
                    
                    if role_ban_giam_hieu:
                        part_response = info_ban_giam_hieu.get(role_ban_giam_hieu)
                        if part_response:
                            responses = part_response
                        else:
                            file_writer = write_file()
                            file_writer.get_ghi_log_file('action_ban_giam_hieu: ' + tracker.latest_message['text'])
                            
                    else:
                        file_writer = write_file()
                        file_writer.get_ghi_log_file('action_ban_giam_hieu: ' + tracker.latest_message['text'])
                    
            
            if responses:
                dispatcher.utter_message(text=responses)
            else:
                dispatcher.utter_message(text=default_response)
        
        
        except Exception as e:
            file_writer = write_file()
            file_writer.get_ghi_log_file(f"Exception in action_thong_tin_dang: {str(e)}")
            dispatcher.utter_message(text=default_response)
        
        return []
       

class ActionHoiDongTruong(Action):
    def name(self):
        return "action_hoi_dong_truong"
   
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
        # Tạo dictionary ánh xạ các role với thông điệp tương ứng
            infor_hoi_dong_truong = constant.get_infor_hoi_dong_truong()

            entities = tracker.latest_message.get('entities', [])
            print(f"\nEntities từ input: \n{entities}")

            default_response = "Xin lỗi tôi không có thông tin về câu hỏi này! Vui lòng cung cấp chi tiết thông tin."
            responses = ""
           
            for entity in entities:
                if entity.get('entity') == 'hoi_dong_truong':
                    role_hoi_dong_truong = entity.get('role')
                    print(f"{role_hoi_dong_truong}")
                    
                    if role_hoi_dong_truong:
                        part_response = infor_hoi_dong_truong.get(role_hoi_dong_truong)
                        if part_response:
                            responses = part_response
                        else:
                            file_writer = write_file()
                            file_writer.get_ghi_log_file('action_thong_tin_dang: ' + tracker.latest_message['text'])
                            
                    else:
                        file_writer = write_file()
                        file_writer.get_ghi_log_file('action_thong_tin_dang: ' + tracker.latest_message['text'])
                    
            
            if responses:
                dispatcher.utter_message(text=responses)
            else:
                dispatcher.utter_message(text=default_response)
        
        except Exception as e:
            file_writer = write_file()
            file_writer.get_ghi_log_file(f"Exception in action_thong_tin_dang: {str(e)}")
            dispatcher.utter_message(text=default_response)
        
        return []


class actionThongTinDang(Action):
    def name(self):
        return "action_thong_tin_dang"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            info_dang = constant.get_infor_dang()
            entities = tracker.latest_message.get('entities', [])
            user_input = tracker.latest_message['text']
            print(f"\n Người dùng hỏi: {user_input}\nEntities từ input: {entities} ")
            
            default_response = "Xin lỗi tôi không có thông tin về câu hỏi này! Vui lòng cung cấp chi tiết thông tin."
            responses = ""
           
            for entity in entities:
                if entity.get('entity') == 'dang_bo_dang_uy':
                    role_dang = entity.get('role')
                    value_dang = entity.get('value')
                    print(f"{role_dang} ----> {value_dang}")
                    
                    if role_dang and value_dang:
                        part_response = info_dang.get(role_dang, {}).get(value_dang)
                        if part_response:
                            responses += part_response
                        else:
                            file_writer = write_file()
                            file_writer.get_ghi_log_file('action_thong_tin_dang: ' + tracker.latest_message['text'])
                            
                    else:
                        file_writer = write_file()
                        file_writer.get_ghi_log_file('action_thong_tin_dang: ' + tracker.latest_message['text'])
                    
            
            if responses:
                dispatcher.utter_message(text=responses)
            else:
                dispatcher.utter_message(text=default_response)
        
        except Exception as e:
            file_writer = write_file()
            file_writer.get_ghi_log_file(f"Exception in action_thong_tin_dang: {user_input}")
            dispatcher.utter_message(text=default_response)
        
        return []


class ActionLanhDaoDangDoanThe(Action):
    def name(self):
        return "action_lanh_dao_dang_doan_the"
   
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        user_input = tracker.latest_message['text']
        print("người dùng hỏi về lãnh đạo: " + user_input)
        # Lấy entities từ câu chat của người dùng
        entities = tracker.latest_message.get('entities', [])
        print(f"\nEntities từ input:\n {entities}")
       
       
        # Tạo dictionary ánh xạ các role với thông điệp tương ứng
        infor_department = {
            'hội đồng trường':'PGS.TS. BS. Nguyễn Minh Phương, Chủ tịch, Email: nmphuong@ctump.edu.vn',
            'đảng và các đoàn thể': "</br>PGS.TS. BS. Nguyễn Minh Phương, Bí thư Đảng ủy, Email: nmphuong@ctump.edu.vn</br>TS. BS. Trần Thanh Hùng, Chủ tịch công đoàn Trường, Email: tthung@ctump.edu.vn</br>TS. BS. Phạm Hoàng Khánh, Bí thư Đoàn TNCSHCM Trường, Email: phkhanh@ctump.edu.vn</br>PGS.TS. BS. Lê Thành Tài, Chủ tịch Hội cựu chiến binh, Email: lttai@ctump.edu.vn</br>",
            "khoa y":"Khoa Y</br>TS.BS. Lê Văn Minh, Trưởng Khoa, Email: lvminh@ctump.edu.vn</br>TS.BS. Trần Thái Thanh Tâm, Phó Trưởng Khoa, Email: ttttam@ctump.edu.vn</br>TS. BS. Nguyễn Như Nghĩa, Phó Trưởng Khoa, Email: nnnghia@ctump.edu.vn</br>",
            "khoa dược":"Khoa Dược</br>PGS.TS. DS Phạm Thành Suôl, Trưởng Khoa, Email: ptsuol@ctump.edu.vn</br>PGS.TS.DS. Đỗ Châu Minh Vĩnh Thọ, Phó Trưởng Khoa, email: dcmvtho@ctump.edu.vn</br>TS. DS Phạm Thị Tố Liên, Phó Trưởng Khoa, Email: pttlien@ctump.edu.vn</br>",
            "khoa răng hàm mặt":"Khoa Răng hàm mặt</br>PGS.TS. BS Trương Nhựt Khuê, Trưởng Khoa, Email: tnkhue@ctump.edu.vn</br>PGS.TS.BS. Lê Nguyên Lâm, Phó Trưởng Khoa, Email: lnlam@ctump.edu.vn</br>TS. BS. Đỗ Thị Thảo, Phó Trưởng Khoa, Email: dtthao@ctump.edu.vn</br>",
            "khoa y tế công cộng":"Khoa Y Tế Công Cộng</br>TS. BS. Lê Minh Hữu, Trưởng Khoa , Email: lmhuu@ctump.edu.vn</br>TS. BS. Nguyễn Tấn Đạt, Phó Trưởng Khoa, Email: ntdat@ctump.edu.vn</br>",
            "khoa điều dưỡng":"Khoa Điều dưỡng</br>TS. BS Nguyễn Hồng Phong, Trưởng Khoa , Email: nhphong@ctump.edu.vn</br>TS. ĐD. Nguyễn Thị Thanh Trúc, Phó Trưởng Khoa, Email: ntttruc@ctump.edu.vn</br>",
            "khoa kỹ thuật y học":"Khoa Kỹ thuật y học</br>TS. BS Nguyễn Hồng Phong, Trưởng Khoa , Email: nhphong@ctump.edu.vn</br>TS. ĐD. Nguyễn Thị Thanh Trúc, Phó Trưởng Khoa, Email: ntttruc@ctump.edu.vn</br>",
            "khoa học cơ bản":"Khoa học cơ bản</br>PGS.TS. Nguyễn Thị Thu Trâm, Trưởng Khoa, Email: ntttram@ctump.edu.vn</br>ThS. Nguyễn Thanh Hùng, Phó Trưởng Khoa, Email: nthung@ctump.edu.vn</br>TS. Trần Thị Hồng Lê, Phó Trưởng Khoa, Email: tthle@ctump.edu.vn</br>",
            "khoa y học cổ truyền":"Khoa Y học cổ truyền</br>TS. BS. Lê Minh Hoàng , Trưởng Khoa, Email: lmhoang@ctump.edu.vn</br>ThS. BS. Nguyễn Ngọc Chi Lan , Phó Trưởng  Khoa, Email: nnclan@ctump.edu.vn</br>",
           
            "phòng thông tin và truyền thông":"Phòng Thông tin và Truyền thông</br>ThS. Châu Minh Khoa, Trưởng Phòng, Email: cmkhoa@ctump.edu.vn</br>ThS. Trần Thị Bích Phương, Phó Trưởng Phòng, Email: ttbphuong@ctump.edu.vn</br>",
            "phòng công tác sinh viên":"Phòng Công tác sinh viên</br>ThS. Ngô Phương Thảo, Trưởng phòng, Email: npthao@ctump.edu.vn</br>ThS. Võ Văn Quyền, Phó Trưởng phòng, Email: vvquyen@ctump.edu.vn</br>",
            "phòng đảm bảo chất lượng":"Phòng đảm bảo chất lượng</br>ThS. BS. Phạm Thị Mỹ Ngọc, Trưởng Phòng, Email: ptmngoc@ctump.edu.vn</br>",
            "phòng đào tạo đại học":"Phòng Đào tạo Đại học</br>TS. BS. Phạm Kiều Anh Thơ, Trưởng phòng, Email: pkatho@ctump.edu.vn</br>TS. BS. Trần Quang Khải, Phó Trưởng phòng, Email: tqkhai@ctump.edu.vn</br>",
            "phòng đào tạo sau đại học":"Phòng Đào tạo Sau đại học</br>GS.TS. BS Võ Huỳnh Trang, Trưởng phòng, Email: vhtrang@ctump.edu.vn</br>TS. BS. Nguyễn Hồng Hà, Phó Trưởng phòng, Email: nhha@ctump.edu.vn</br>TS. DS Đặng Duy Khánh, Phó Trưởng phòng , Email: ddkhanh@ctump.edu.vn</br>",
            "phòng hành chính tổng hợp":"Phòng Hành chính tổng hợp</br>ThS. Phạm Trương Yến Nhi , Trưởng phòng, Email: ptynhi@ctump.edu.vn</br>ThS. Phạm Thị Minh, Phó Trưởng phòng, Email: ptminh@ctump.edu.vn</br>CN. Nguyễn Văn Bộ, Phó Trưởng phòng, Email: nvbo@ctump.edu.vn</br>",
            "phòng khảo thi":"Phòng Khảo thí</br>ThS. Phan Thị Tuyết Nhung, Trưởng Phòng, Email: pttnhung@ctump.edu.vn</br>Ths. DS. Lê Thị Minh Ngọc, Phó Trưởng Phòng, Email: ltmngoc@ctump.edu.vn</br>",
            "phòng khoa học công nghệ và quan hệ đối ngoại":"Phòng Khoa học công nghệ và Quan hệ đối ngoại</br>PGS. TS. DS. Nguyễn Thị Ngọc Vân, Trưởng phòng, Email:  ntnvan@ctump.edu.vn</br>PGS.TS. DS. Nguyễn Thắng, Phó Trưởng phòng, Email: nthang@ctump.edu.vn</br>TS. BS. Hoàng Minh Tú, Phó Trưởng phòng, Email: hmtu@ctump.edu.vn</br>",
            "phòng quản trị thiết bị":"Phòng Quản trị thiết bị</br>ThS. Nguyễn Văn Tám, Trưởng phòng, Email: nvtam@ctump.edu.vn</br>ThS. Huỳnh Trường Hiệp, Phó Trưởng phòng, Email: hthiep@ctump.edu.vn</br>",
            "phòng tài chính kế toán":"Phòng Tài chính kế toán</br>ThS. Võ Nhật Nhân Tuyền, Trưởng phòng, Email: vnntuyen@ctump.edu.vn</br>CN. Hứa Kim Chi, Phó Trưởng phòng, Email: hkchi@ctump.edu.vn</br>",
            "phòng tổ chức cán bộ":"Phòng Tổ chức Cán bộ</br>ThS. Trần Trương Ngọc Bích, Trưởng Phòng, Email:  ttnbich@ctump.edu.vn</br>ThS. Hà Bảo Trân, Phó Trưởng phòng, Email: hbtran@ctump.edu.vn</br>",
            "phòng thanh tra pháp chế":"Phòng Thanh tra - Pháp chế</br>TS. BS. Bùi Quang Nghĩa, Trưởng Phòng, Email: bqnghia@ctump.edu.vn</br>CN. Nguyễn Hiệp Phúc, Phó Trưởng phòng, Email: nhphuc@ctump.edu.vn</br>",
            "ban quản lý dự án":"Ban Quản lý dự án</br>KS. Nguyễn Chí Trung, Phó Trưởng Ban, Email: nctrung@ctump.edu.vn</br>",
            "trung tâm dịch vụ và đào tạo theo nhu cầu xã hội":"Trung tâm Dịch vụ và Đào tạo theo nhu cầu xã hội</br>TS. BS. Nguyễn Triều Việt, Giám đốc, Email: ntviet@ctump.edu.vn</br>TS. Phan Thị Luyện, Phó Giám đốc, Email: ptluyen@ctump.edu.vn</br>",
            "trung tâm giáo dục y học và huấn luyện kỹ năng y khoa":"Trung tâm Giáo dục y học và Huấn luyện kỹ năng y khoa</br>ThS. BS Phạm Thị Mỹ Ngọc, Trưởng Trung tâm, Email: ptmngoc@ctump.edu.vn</br>ThS. BS. Trần Xuân Quỳnh, Phó Trưởng Trung tâm, Email: txquynh@ctump.edu.vn</br>",
            "thư viện":"Thư viện</br>TS. BS. Nguyễn Thị Hải Yến, Giám đốc, Email: nthyen@ctump.edu.vn</br>TS.BS. Lê Thị Hoàng Mỹ , Phó Giám đốc, Email: lthmy@ctump.edu.vn</br>",
            "bệnh viện":"BỆNH VIỆN</br>BSCKII. Lại Văn Nông, Giám đốc, Email: lvnong@ctump.edu.vn.</br>PGS. TS.BS. Nguyễn Thành Tấn, Phó Giám đốc, Email: nttan@ctump.edu.vn</br>ThS. Nguyễn Văn Tám, Phó Giám đốc, Email: nvtam@ctump.edu.vn</br>TS. BS. Võ Phạm Minh Thư, Phó Giám đốc, Email: vpmthu@ctump.edu.vn</br>"
           


        }
       
        input_value = None
       
        for entity in entities:
            if entity.get('entity') == "department":
                input_value = entity.get('value')
                print("----"+input_value)
        if input_value in infor_department:
            response = infor_department[input_value]
        else:
            file_writer = write_file()
            file_writer.get_ghi_log_file('action_lanh_dao_dang_doan_the: '+ tracker.latest_message['text'])
            response= "Xin lỗi, tôi không có thông tin về câu hỏi của bạn."
        dispatcher.utter_message(text=response)
        return []


   

class actionThongTinKhoa(Action):
    def name(self):
        return "action_thong_tin_khoa"
    def run(self, dispatcher: CollectingDispatcher,tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        list_thong_tin = constant.get_nganh_hoc()
        
        #lấy ra entity từ câu chat
        entities = tracker.latest_message['entities']
        print(f"\n\nEntities từ input: {entities} ")
        
        
        khoa_phong = None
        
        for entity in entities:
            if entity.get('entity') == 'khoa_phong':
                khoa_phong = entity.get('value')
                print("value:", khoa_phong)
            
        if khoa_phong in list_thong_tin:
            khoa_info = list_thong_tin[khoa_phong]
            response = f"Thông tin về {khoa_info['Tên']}:"
            response += f"Địa chỉ: {khoa_info['Địa chỉ']}"
            response += f"Số điện thoại: {khoa_info['Điện thoại']}"
            response += f"Email: {khoa_info['Email']}"
        else:
            file_writer = write_file()
            file_writer.get_ghi_log_file('action_cap_lai_the_sv: '+ tracker.latest_message['text'])
            response = "Xin lỗi tôi không có thông tin về câu hỏi này! Vui lòng cung cấp chi tiết thông tin."
                               
        dispatcher.utter_message(text=response)
        
        return [SlotSet("ask", None) , SlotSet("khoa_phong", None)]
    
class actionGiayxacnhan(Action):
    def name(self):
        return "action_giay_xac_nhan"
    def run(self, dispatcher: CollectingDispatcher,tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # lấy entities từ câu chat của người dùng
        entities = tracker.latest_message['entities']
        print(f"\n\nEntities từ input: {entities} ")
        
        #lấy ra slots
        giayxacnhan = tracker.get_slot('giayxacnhan')
        ask = tracker.get_slot('ask')
        print(f"Slot giayxacnhan: {giayxacnhan}")
        print(f"Slot ask: {ask}")
        
        #mảng role 
        giayxacnhan_roles = {
            'GiayVayVon': False,
            'GiayTamHoanNghiaVu': False,
            'GiamThue': False
        }

        list_thong_tin_can_biet = constant.get_thong_tin_can_biet()

        # Phân loại các role từ entities
        for entity in entities:
            if entity.get('entity') == 'giayxacnhan' and entity.get('role'):
                role = entity.get('role')
                if role in giayxacnhan_roles:
                    giayxacnhan_roles[role] = True
        print(f"Roles của giayxacnhan: {giayxacnhan_roles}")

        if giayxacnhan_roles['GiayVayVon'] or  giayxacnhan_roles['GiayTamHoanNghiaVu'] or giayxacnhan_roles['GiamThue']:
            thong_tin = list_thong_tin_can_biet['hoan_nghia_vu_vay_von_giam_thue']
            response = f"{thong_tin['mota']}"
            response += f"Yêu cầu:{thong_tin['yeucau']}"
            response += f"{thong_tin['thoigian']}"
        else:
            file_writer = write_file()
            file_writer.get_ghi_log_file('action_cap_lai_the_sv: '+ tracker.latest_message['text'])
            response = "Xin lỗi tôi không có thông tin về câu hỏi này! Vui lòng cung cấp chi tiết thông tin."

        # Gửi phản hồi tới người dùng
        dispatcher.utter_message(text=response)
        return [SlotSet("ask", None) , SlotSet("giayxacnhan", None)]

class actionTroCap(Action):
    def name(self):
        return "action_tro_cap"
    def run(self, dispatcher: CollectingDispatcher,tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        #lấy ra entity từ câu chat
        entities = tracker.latest_message['entities']
        print(f"\n\nEntities từ input: {entities}")
        
        tro_cap_slot = tracker.get_slot('tro_cap')
        print(f"Slot trợ cấp: {tro_cap_slot}")
        
        list_thong_tin_can_biet = constant.get_thong_tin_can_biet()
        
        input_value = None
        
        for entity in entities:
            if entity.get('entity') == "tro_cap":
                input_value = entity.get('value')
                
        if input_value in list_thong_tin_can_biet:
            thong_tin = list_thong_tin_can_biet[input_value]
            response = f"{thong_tin['mota']}"
            response += f"Yêu cầu:{thong_tin['yeucau']}"
            response += f"{thong_tin['thoigian']}"
        else:
            file_writer = write_file()
            file_writer.get_ghi_log_file('action_cap_lai_the_sv: '+ tracker.latest_message['text'])
            response= "Xin lỗi tôi không có thông tin về câu hỏi này! Vui lòng cung cấp chi tiết thông tin."
        dispatcher.utter_message(text=response)
        return []

class actionThuVien_TTHL(Action):
    def name(self):
        return "action_thuvien_tthl"
    def run(self, dispatcher: CollectingDispatcher,tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        entities = tracker.latest_message['entities']
        print(f"Entity từ input: {entities}")
        
        list_thong_tin_can_biet = constant.get_thong_tin_can_biet()
        
        the_thu_vien_slot = tracker.get_slot('the_tv_tthl')
        print(f"Slot thẻ thư viện/TTHL: {the_thu_vien_slot}")
        
        input_value = None
        
        for entity in entities:
            if (entity.get('entity') == "the_tv_tthl"):
                input_value = entity.get('value')
                
        if input_value in list_thong_tin_can_biet:
            thong_tin = list_thong_tin_can_biet[input_value]
            response = f"{thong_tin['mota']}"
            response += f"Yêu cầu:{thong_tin['yeucau']}"
            response += f"{thong_tin['thoigian']}"
        else:
            file_writer = write_file()
            file_writer.get_ghi_log_file('action_cap_lai_the_sv: '+ tracker.latest_message['text'])
            response = "Xin lỗi tôi không có thông tin về câu hỏi này! Vui lòng cung cấp chi tiết thông tin."

            
        dispatcher.utter_message(text= response)
        return []

class actionGiayCongTac(Action):
    def name(self):
        return "action_giay_cong_tac"
    
    def run(self, dispatcher: CollectingDispatcher,tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        list_thong_tin_can_biet = constant.get_thong_tin_can_biet()
        
        entities = tracker.latest_message['entities']
        print(f"Entity từ input: {entities}")

        slot_giay_cong_tac = tracker.get_slot('giay_cong_tac')
        print(f"Slot giay cong tac {slot_giay_cong_tac}")

        input_value = None

        for entity in entities:
            if entity.get('entity') == "giay_CT":
                input_value = entity.get('value')
                
        if input_value in list_thong_tin_can_biet:
            thong_tin = list_thong_tin_can_biet[input_value]
            response = f"{thong_tin['mota']}"
            response += f"Yêu cầu:{thong_tin['yeucau']}"
            response += f"{thong_tin['thoigian']}"
        else:
            file_writer = write_file()
            file_writer.get_ghi_log_file('action_cap_lai_the_sv: '+ tracker.latest_message['text'])
            response = "Xin lỗi tôi không có thông tin về câu hỏi này! Vui lòng cung cấp chi tiết thông tin."

        dispatcher.utter_message(text=response)
        return []

class actionCapTheSinhVien(Action):
    def name(self):
        return "action_cap_the_sv"
    def run(self, dispatcher: CollectingDispatcher,tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        list_thong_tin = constant.get_thong_tin_can_biet()
        
        entities = tracker.latest_message['entities']
        print(f"\n\nEntity từ input : {entities}")
        
        input_value = None
        
        for entity in entities:
            if entity.get('entity') == "the_sv":
                input_value = entity.get('value')
                
        if input_value in list_thong_tin:
            thong_tin = list_thong_tin[input_value]
            response = f"{thong_tin['mota']}"
            response += f"Yêu cầu:{thong_tin['yeucau']}"
            response += f"{thong_tin['thoigian']}"
        else: 
            file_writer = write_file()
            file_writer.get_ghi_log_file('action_cap_lai_the_sv: '+ tracker.latest_message['text'])
            response = "Xin lỗi tôi không có thông tin về câu hỏi này! Vui lòng cung cấp chi tiết thông tin."
        
        dispatcher.utter_message(text = response)
        return []
    
class actionHocBong(Action):
    def name(self):
        return "action_hoc_bong"
    def run(self, dispatcher: CollectingDispatcher,tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        list_thong_tin = constant.get_thong_tin_can_biet()
        
        entities = tracker.latest_message['entities']
        print(f"\n\nEntity từ input : {entities}")
        
        input_value = None
        
        for entity in entities:
            if entity.get('entity') == "hoc_bong":
                input_value = entity.get('value')
                
        if input_value in list_thong_tin:
            thong_tin = list_thong_tin[input_value]
            response = f"{thong_tin['mota']}"
            response += f"Yêu cầu: {thong_tin['yeucau']}"
            response += f"{thong_tin['thoigian']}"
        else: 
            file_writer = write_file()
            file_writer.get_ghi_log_file('action_cap_lai_the_sv: '+ tracker.latest_message['text'])
            response = "Xin lỗi tôi không có thông tin về câu hỏi này! Vui lòng cung cấp chi tiết thông tin."
        
        dispatcher.utter_message(text = response)
        return []
    
class actionBaoLuu(Action):
    def name(self):
        return "action_baoluu_thoihoc"
    def run(self, dispatcher: CollectingDispatcher,tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        list_thong_tin = constant.get_thong_tin_can_biet()
        
        entities = tracker.latest_message['entities']
        print(f"\n\nEntity từ input : {entities}")
        
        thoihoc_slot = tracker.get_slot('baoluu_thoihoc')
        print(f"slot của baoluu_thoihoc là: {thoihoc_slot}")
        
        input_value = None
        
        for entity in entities:
            if entity.get('entity') == "baoluu_thoihoc":
                input_value = entity.get('value')
                
        if input_value in list_thong_tin:
            thong_tin = list_thong_tin[input_value]
            response = f"{thong_tin['mota']}"
            response += f"Yêu cầu: {thong_tin['yeucau']}"
            response += f"{thong_tin['thoigian']}"
        else: 
            file_writer = write_file()
            file_writer.get_ghi_log_file('action_cap_lai_the_sv: '+ tracker.latest_message['text'])
            response = "Xin lỗi tôi không có thông tin về câu hỏi này! Vui lòng cung cấp chi tiết thông tin."
        
        dispatcher.utter_message(text = response)
        return []

class actionBHYT(Action):
    def name(self):
        return "action_cap_thaydoi_BHYT"
    def run(self, dispatcher: CollectingDispatcher,tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        list_thong_tin = constant.get_thong_tin_can_biet()
        
        entities = tracker.latest_message['entities']
        print(f"\n\nEntity từ input : {entities}")
        
        BHYT_slot = tracker.get_slot('the_BHYT')
        print(f"slot của the_BHYT là: {BHYT_slot}")
        
        input_value = None
        
        for entity in entities:
            if entity.get('entity') == "the_BHYT":
                input_value = 'caplai_thaydoi_BHYT'
                
        if input_value in list_thong_tin:
            thong_tin = list_thong_tin[input_value]
            response = f"{thong_tin['mota']}"
            response += f"Yêu cầu: {thong_tin['yeucau']}"
            response += f"{thong_tin['thoigian']}"
        else: 
            file_writer = write_file()
            file_writer.get_ghi_log_file('action_cap_lai_the_sv: '+ tracker.latest_message['text'])
            response = "Xin lỗi tôi không có thông tin về câu hỏi này! Vui lòng cung cấp chi tiết thông tin."
        
        dispatcher.utter_message(text = response)
        return []
    
class actionTK_QLDT(Action):
    def name(self):
        return "action_caplai_tk_QLDT"
    def run(self, dispatcher: CollectingDispatcher,tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        list_thong_tin = constant.get_thong_tin_can_biet()
        
        entities = tracker.latest_message['entities']
        print(f"\n\nEntity từ input : {entities}")
        
        QLDT_slot = tracker.get_slot('taikhoan_QLDT')
        print(f"slot của the_BHYT là: {QLDT_slot}")
        
        input_value = None
        
        for entity in entities:
            if entity.get('entity') == "tk_qldt":
                input_value = 'caplai_qldt'
                
        if input_value in list_thong_tin:
            thong_tin = list_thong_tin[input_value]
            response = f"{thong_tin['mota']}"
            response += f"Yêu cầu: {thong_tin['yeucau']}"
            response += f"{thong_tin['thoigian']}"
        else: 
            file_writer = write_file()
            file_writer.get_ghi_log_file('action_cap_lai_the_sv: '+ tracker.latest_message['text'])
            response = "Xin lỗi tôi không có thông tin về câu hỏi này! Vui lòng cung cấp chi tiết thông tin."
        
        dispatcher.utter_message(text = response)
        return []

class actionEmailSV(Action):
    def name(self):
        return "action_caplai_email"
    def run(self, dispatcher: CollectingDispatcher,tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        list_thong_tin = constant.get_thong_tin_can_biet()
        
        entities = tracker.latest_message['entities']
        print(f"\n\nEntity từ input : {entities}")
        
        input_value = None
        
        email_sv_slot = tracker.get_slot('email_sv')
        print(f"slot của caplai_email là: {email_sv_slot}")
        
        for entity in entities:
            if entity.get('entity') == "caplai_email_sv":
                input_value = 'caplai_email'
                
        if input_value in list_thong_tin:
            thong_tin = list_thong_tin[input_value]
            response = f"{thong_tin['mota']}"
            response += f"Yêu cầu: {thong_tin['yeucau']}"
            response += f"{thong_tin['thoigian']}"
        else: 
            file_writer = write_file()
            file_writer.get_ghi_log_file('action_cap_lai_the_sv: '+ tracker.latest_message['text'])
            response = "Xin lỗi tôi không có thông tin về câu hỏi này! Vui lòng cung cấp chi tiết thông tin."
        
        dispatcher.utter_message(text = response)
        return []
    
class actionBHTN(Action):
    def name(self):
        return "action_caplai_thanhphi_BHTN"
    def run(self, dispatcher: CollectingDispatcher,tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        list_thong_tin = constant.get_thong_tin_can_biet()
        
        entities = tracker.latest_message['entities']
        print(f"\n\nEntity từ input : {entities}")
        
        input_value = None
        
        caplai_thanhphi_BHTN_slot = tracker.get_slot('caplai_thanhphi_BHTN')
        print(f"slot của caplai_thanhphi_bhtn là: {caplai_thanhphi_BHTN_slot}")
        
        for entity in entities:
            if entity.get('entity') == "caplai_thanhphi_bhtn":
                input_value = 'caplai_thanhphi_BHTN'
                
        if input_value in list_thong_tin:
            thong_tin = list_thong_tin[input_value]
            response = f"{thong_tin['mota']}"
            response += f"Yêu cầu: {thong_tin['yeucau']}"
            response += f"{thong_tin['thoigian']}"
        else: 
            file_writer = write_file()
            file_writer.get_ghi_log_file('action_cap_lai_the_sv: '+ tracker.latest_message['text'])
            response = "Xin lỗi tôi không có thông tin về câu hỏi này! Vui lòng cung cấp chi tiết thông tin."
        
        dispatcher.utter_message(text = response)
        return []
    
 
# class actionHocPhi(Action):
#     def name(self):
#         return "action_hocphi_nganh"
#     def run(self, dispatcher: CollectingDispatcher,tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
#         entities = tracker.latest_message['entities']
#         print(f"\n\nEntity từ input : {entities}")
#         infor_ctdt_hoc_phi = constant.get_hocphi_ctdt()
#         response = "Xin lỗi tôi không có thông tin về câu hỏi này! Vui lòng cung cấp chi tiết thông tin."
#         role = []
#         slot_ten_nganh = tracker.get_slot('hocphi_nganh')
#         print(slot_ten_nganh)
#         for entity in entities:
#             if entity.get('role') not in role:
#                 role.append(entity.get('role'))
#         print("mảng role: ", role)
        
#         if 'ten_nganh' in role and 'hoi_hocphi' in role:
#             hocphi_chinhquy = ""
#             hocphi_theonhucauxh = ""
#             hocphi_cutuyen = ""
#             hocphi_qt_tiengviet = ""

#             hocphi_lt_chinhquy = ""
#             hocphi_lt_theonhucauxh = ""

#             trungcap_dh_chinhquy = ""
#             trungcap_dh_nhu_cau_xh = ""
#             caodang_dh_chinhquy = ""
#             caodang_dh_nhucau_xh = ""

#             # Chính Quy 
#             if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['chinhquy']:
#                 hocphi_chinhquy = infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['chinhquy'][slot_ten_nganh]['hocphi'] + "/tín chỉ"
#             else : hocphi_chinhquy = ""

#             if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['nhu_cau_xh']:
#                 hocphi_theonhucauxh = infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['nhu_cau_xh'][slot_ten_nganh]['hocphi'] + "/tín chỉ"
#             else : hocphi_theonhucauxh = ""

#             if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['cu_tuyen']:
#                 hocphi_cutuyen = infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['cu_tuyen'][slot_ten_nganh]['hocphi'] + "/tín chỉ"
#             else : hocphi_cutuyen = ""

#             if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['qt_tiengviet']:
#                 hocphi_qt_tiengviet = infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['qt_tiengviet'][slot_ten_nganh]['hocphi'] + "/tín chỉ"
#             else : hocphi_qt_tiengviet = ""

#             # Liên thông
#             if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_lienthong']['lt_chinhquy']:
#                 hocphi_lt_chinhquy = infor_ctdt_hoc_phi['daihoc']['he_lienthong']['lt_chinhquy'][slot_ten_nganh]['hocphi'] + "/tín chỉ"
#             else : hocphi_lt_chinhquy = ""

#             if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_lienthong']['lt_nhu_cau_xh']:
#                 hocphi_lt_theonhucauxh = infor_ctdt_hoc_phi['daihoc']['he_lienthong']['lt_nhu_cau_xh'][slot_ten_nganh]['hocphi'] + "/tín chỉ"
#             else : hocphi_lt_theonhucauxh = ""

#             # Vừa học vừa làm
#             if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_vuahoc_vualam']['trungcap_dh_chinhquy']:
#                 trungcap_dh_chinhquy = infor_ctdt_hoc_phi['daihoc']['he_vuahoc_vualam']['trungcap_dh_chinhquy'][slot_ten_nganh]['hocphi'] + "/tín chỉ"
#             else : trungcap_dh_chinhquy = ""

#             if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_vuahoc_vualam']['trungcap_dh_nhu_cau_xh']:
#                 trungcap_dh_nhu_cau_xh = infor_ctdt_hoc_phi['daihoc']['he_vuahoc_vualam']['trungcap_dh_nhu_cau_xh'][slot_ten_nganh]['hocphi'] + "/tín chỉ"
#             else : trungcap_dh_nhu_cau_xh = ""

#             if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_vuahoc_vualam']['caodang_dh_chinhquy']:
#                 caodang_dh_chinhquy = infor_ctdt_hoc_phi['daihoc']['he_vuahoc_vualam']['caodang_dh_chinhquy'][slot_ten_nganh]['hocphi'] + "/tín chỉ"
#             else : caodang_dh_chinhquy = ""

#             if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_vuahoc_vualam']['caodang_dh_nhucau_xh']:
#                 caodang_dh_nhucau_xh = infor_ctdt_hoc_phi['daihoc']['he_vuahoc_vualam']['caodang_dh_nhucau_xh'][slot_ten_nganh]['hocphi'] + "/tín chỉ"
#             else : caodang_dh_nhucau_xh = ""
            
#             #Chính quy
#             response = f"Học phí đào tạo đại học của {slot_ten_nganh}:"
#             response += f"<br>- Hệ chính quy: {hocphi_chinhquy}"
#             response += f"<br>- Hệ theo nhu cầu xã hội: {hocphi_theonhucauxh}"
#             response += f"<br>- Hệ cử tuyển: {hocphi_cutuyen}"
#             response += f"<br>- Hệ quốc tế tiếng việt: {hocphi_qt_tiengviet}"

#             #Liên thông
#             response += f"<br>- Hệ liên thông chính quy: {hocphi_lt_chinhquy}"
#             response += f"<br>- Hệ liên thông theo nhu cầu xã hội: {hocphi_lt_theonhucauxh}"

#             # Vừa học vừa làm
#             response += f"<br>- Hệ vừa học vừa làm:<br>+ Trung cấp đào tạo lên đại học (chính quy): {trungcap_dh_chinhquy}"
#             response += f"<br>+ Trung cấp đào tạo lên đại học (theo nhu cầu XH): {trungcap_dh_nhu_cau_xh}"
#             response += f"<br>+ Cao đẳng đào tạo lên đại học (chính quy): {caodang_dh_chinhquy}"
#             response += f"<br>+ Cao đăng đào tạo lên đại học (theo nhu cầu XH): {caodang_dh_nhucau_xh}"    
#         else: response = f"Xin lỗi, tôi không tìm thấy thông tin học phí cho {slot_ten_nganh}."
        
#         if  'hoi_hocphi' in role and 'ten_nganh' not in role:
#             response = "Tôi có thông tin học phí về các ngành của các chương trình đào tạo, bạn cần biết học phí ngành nào?"

#         dispatcher.utter_message(text=response)
        
#         return [SlotSet("hocphi_nganh", None)]
    

class actionHocPhi(Action):
    def name(self):
        return "action_hocphi_nganh"
    def run(self, dispatcher: CollectingDispatcher,tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        print("ACTION HOI HOC PHI")
        
        entities = tracker.latest_message['entities']
        print(f"\n\nEntity từ input : {entities}")
        infor_ctdt_hoc_phi = constant.get_hocphi_ctdt()
        role = []
        slot_ten_nganh = tracker.get_slot('hocphi_nganh')
        print(f"Slot tên ngành là : {slot_ten_nganh}")
        for entity in entities:
            if entity.get('role') not in role:
                role.append(entity.get('role'))
        print("mảng role: ", role)
        response = "Bạn vui lòng nhập đầy đủ câu hỏi<br> Ví dụ:<br>- Học phí + tên ngành<br>- Chương trình đào tạo + tên ngành"
        found = False
        
        if 'ten_nganh' in role and 'hoi_hocphi' in role:
            response = f"<br>Học phí của {slot_ten_nganh} : "
            print("Học phí của ngành là")
            # Chính Quy 
            #Trúng tuyển chính quy
            if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['chinhquy']:
                print("Học phí chính quy")
                response += "<br>Hệ chính quy :" 
                response += f"<br>- Đào tạo trúng tuyển chính quy : " + infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['chinhquy'][slot_ten_nganh]['hocphi'] + "/tín chỉ"
                found = True
            #đào tạo theo nhu cầu xã hội
            if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['nhu_cau_xh']:
                response += "<br>- Đào tạo theo nhu cầu xã hội (theo đặt hàng ) : " + infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['nhu_cau_xh'][slot_ten_nganh]['hocphi'] + "/tín chỉ"
                found = True
            #đào tạo hệ cử tuyển
            if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['cu_tuyen']:
                response += "<br>- Đào tạo hệ cử tuyển : "+ infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['cu_tuyen'][slot_ten_nganh]['hocphi'] + "/tín chỉ"
                found = True
            #đào tạo sinh viên quốc tế chương trình tiếng việt
            if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['qt_tiengviet']:
                response +="<br>- Đào tạo sinh viên quốc tế chương trình tiếng Việt: " + infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['qt_tiengviet'][slot_ten_nganh]['hocphi'] + "/tín chỉ"
                found = True
            # Liên thông
            # Liên thông chính quy
            if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_lienthong']['lt_chinhquy']:
                response += "<br>Hệ liên thông : "
                response += "<br>- Đào tạo liên thông hệ chính quy : " + infor_ctdt_hoc_phi['daihoc']['he_lienthong']['lt_chinhquy'][slot_ten_nganh]['hocphi'] + "/tín chỉ"
                found = True
            #Liên thông theo nhu cầu xã hội
            if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_lienthong']['lt_nhu_cau_xh']:
                response += "<br>- Đào tạo liên thông theo nhu cầu xã hội : " + infor_ctdt_hoc_phi['daihoc']['he_lienthong']['lt_nhu_cau_xh'][slot_ten_nganh]['hocphi'] + "/tín chỉ"
                found = True
            # Vừa học vừa làm
            if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_vuahoc_vualam']['trungcap_dh_chinhquy']:
                response += "<br>Hệ liên thông vừa học vừa làm :"
                response +="<br>- Hệ trung cấp đầo tạo lên đại học (theo chương trình chính quy): " + infor_ctdt_hoc_phi['daihoc']['he_vuahoc_vualam']['trungcap_dh_chinhquy'][slot_ten_nganh]['hocphi'] + "/tín chỉ"
                found = True
            if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_vuahoc_vualam']['trungcap_dh_nhu_cau_xh']:
                response += "<br>- Hệ trung cấp đào tạo lên đại học (theo nhu cầu xã hội): " + infor_ctdt_hoc_phi['daihoc']['he_vuahoc_vualam']['trungcap_dh_nhu_cau_xh'][slot_ten_nganh]['hocphi'] + "/tín chỉ"
                found = True
            if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_vuahoc_vualam']['caodang_dh_chinhquy']:
                response += "<br>- Hệ cao đẳng đào tạo lên đại học (theo chương trình chính quy): " + infor_ctdt_hoc_phi['daihoc']['he_vuahoc_vualam']['caodang_dh_chinhquy'][slot_ten_nganh]['hocphi'] + "/tín chỉ"
                found = True
            if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_vuahoc_vualam']['caodang_dh_nhucau_xh']:
                response += "<br>- Hệ cao đẳng đào tạo lên đại học (theo nhu cầu xã hội): " + infor_ctdt_hoc_phi['daihoc']['he_vuahoc_vualam']['caodang_dh_nhucau_xh'][slot_ten_nganh]['hocphi'] + "/tín chỉ"
                found = True
            if found == False:
                response = f"Không tìm thấy ngành bạn yêu cầu."
        
        dispatcher.utter_message(text=response)
        
        return [SlotSet("hocphi_nganh", None)]
    

class actionGoiYHocPhi(Action):
    def name(self):
        return "action_goi_y_hocphi"
    def run(self, dispatcher: CollectingDispatcher,tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        print("action Gợi ý học phí")
        response = "Tôi có thông tin học phí của các ngành:<br>- Y khoa,<br>- Răng hàm mặt,<br>- Y học dự phòng<br>- Y học cổ truyền<br>- Dược<br>- Điều dưỡng<br>- Y tế công cộng<br>- Kỹ thuật xét nghiệm y học<br>- Hộ sinh<br>- Kỹ thuật hình ảnh y học"

        dispatcher.utter_message(text=response)
        
        return [SlotSet("hocphi_nganh", None)]
    

class actionGoiYCTDT(Action):
    def name(self):
        return "action_goi_y_CTDT"
    def run(self, dispatcher: CollectingDispatcher,tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print("action Gợi ý ctdt")
        response = "Tôi có thông tin chương trình đào tạo của các ngành:<br>- Y khoa,<br>- Răng hàm mặt,<br>- Y học dự phòng<br>- Y học cổ truyền<br>- Dược<br>- Điều dưỡng<br>- Y tế công cộng<br>- Kỹ thuật xét nghiệm y học<br>- Hộ sinh<br>- Kỹ thuật hình ảnh y học"

        dispatcher.utter_message(text=response)
        
        return []


# class actionCTDT(Action):
#     def name(self):
#         return "action_CTDT_nganh"
#     def run(self, dispatcher: CollectingDispatcher,tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
       
#         entities = tracker.latest_message['entities']
#         print(f"\n\nEntity từ input : {entities}")
#         infor_ctdt_hoc_phi = constant.get_hocphi_ctdt()
#         response = "Bạn vui lòng nhập đầy đủ câu hỏi<br> Ví dụ:<br>- Chương trình đào tạo + tên ngành<br>- Chương trình đào tạo + tên ngành"
#         role = []
#         slot_ten_nganh = tracker.get_slot('CTDT_nganh')
#         print('action ctdt ngành')
#         print(slot_ten_nganh)
#         for entity in entities:
#             if entity.get('role') not in role:
#                 role.append(entity.get('role'))
#         print("mảng role: ", role)
#         if slot_ten_nganh not in infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['chinhquy'] and \
#            slot_ten_nganh not in infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['nhu_cau_xh'] and \
#            slot_ten_nganh not in infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['cu_tuyen'] and \
#            slot_ten_nganh not in infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['qt_tiengviet'] and \
#            slot_ten_nganh not in infor_ctdt_hoc_phi['daihoc']['he_lienthong']['lt_chinhquy'] and \
#            slot_ten_nganh not in infor_ctdt_hoc_phi['daihoc']['he_lienthong']['lt_nhu_cau_xh'] and \
#            slot_ten_nganh not in infor_ctdt_hoc_phi['daihoc']['he_vuahoc_vualam']['trungcap_dh_chinhquy'] and \
#            slot_ten_nganh not in infor_ctdt_hoc_phi['daihoc']['he_vuahoc_vualam']['trungcap_dh_nhu_cau_xh'] and \
#            slot_ten_nganh not in infor_ctdt_hoc_phi['daihoc']['he_vuahoc_vualam']['caodang_dh_chinhquy'] and \
#            slot_ten_nganh not in infor_ctdt_hoc_phi['daihoc']['he_vuahoc_vualam']['caodang_dh_nhucau_xh']:
#             response = f"Tôi không tìm thấy thông tin về ngành học theo yêu cầu."
#         else:
#             if 'ten_nganh' in role and 'hoi_ctdt' in role:
#                 ctdt_chinhquy = ""
#                 ctdt_theonhucauxh = ""
#                 ctdt_cutuyen = ""
#                 ctdt_qt_tiengviet = ""

#                 ctdt_lt_chinhquy = ""
#                 ctdt_lt_theonhucauxh = ""

#                 trungcap_dh_chinhquy = ""
#                 trungcap_dh_nhu_cau_xh = ""
#                 caodang_dh_chinhquy = ""
#                 caodang_dh_nhucau_xh = ""

#                 try:                
#                     # Chính Quy 
#                     if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['chinhquy']:
#                         ctdt_chinhquy = infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['chinhquy'][slot_ten_nganh]['ctdt'] 
#                     else : ctdt_chinhquy = "Chưa cập nhật"

#                     if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['nhu_cau_xh']:
#                         ctdt_theonhucauxh = infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['nhu_cau_xh'][slot_ten_nganh]['ctdt'] 
#                     else : ctdt_theonhucauxh = "Chưa cập nhật"

#                     if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['cu_tuyen']:
#                         ctdt_cutuyen = infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['cu_tuyen'][slot_ten_nganh]['ctdt'] 
#                     else : ctdt_cutuyen = "Chưa cập nhật"

#                     if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['qt_tiengviet']:
#                         ctdt_qt_tiengviet = infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['qt_tiengviet'][slot_ten_nganh]['ctdt'] 
#                     else : ctdt_qt_tiengviet = "Chưa cập nhật"

#                     # Liên thông
#                     if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_lienthong']['lt_chinhquy']:
#                         ctdt_lt_chinhquy = infor_ctdt_hoc_phi['daihoc']['he_lienthong']['lt_chinhquy'][slot_ten_nganh]['ctdt'] 
#                     else : ctdt_lt_chinhquy = "Chưa cập nhật"

#                     if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_lienthong']['lt_nhu_cau_xh']:
#                         ctdt_lt_theonhucauxh = infor_ctdt_hoc_phi['daihoc']['he_lienthong']['lt_nhu_cau_xh'][slot_ten_nganh]['ctdt'] 
#                     else : ctdt_lt_theonhucauxh = "Chưa cập nhật"

#                     # Vừa học vừa làm
#                     if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_vuahoc_vualam']['trungcap_dh_chinhquy']:
#                         trungcap_dh_chinhquy = infor_ctdt_hoc_phi['daihoc']['he_vuahoc_vualam']['trungcap_dh_chinhquy'][slot_ten_nganh]['ctdt'] 
#                     else : trungcap_dh_chinhquy = "Chưa cập nhật"

#                     if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_vuahoc_vualam']['trungcap_dh_nhu_cau_xh']:
#                         trungcap_dh_nhu_cau_xh = infor_ctdt_hoc_phi['daihoc']['he_vuahoc_vualam']['trungcap_dh_nhu_cau_xh'][slot_ten_nganh]['ctdt'] 
#                     else : trungcap_dh_nhu_cau_xh = "Chưa cập nhật"

#                     if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_vuahoc_vualam']['caodang_dh_chinhquy']:
#                         caodang_dh_chinhquy = infor_ctdt_hoc_phi['daihoc']['he_vuahoc_vualam']['caodang_dh_chinhquy'][slot_ten_nganh]['ctdt'] 
#                     else : caodang_dh_chinhquy = "Chưa cập nhật"

#                     if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_vuahoc_vualam']['caodang_dh_nhucau_xh']:
#                         caodang_dh_nhucau_xh = infor_ctdt_hoc_phi['daihoc']['he_vuahoc_vualam']['caodang_dh_nhucau_xh'][slot_ten_nganh]['ctdt'] 
#                     else : caodang_dh_nhucau_xh = "Chưa cập nhật"
                    
#                     #Chính quy
#                     response = f"Chương trình đào tạo đại học của {slot_ten_nganh}:"
#                     response += f"<br>- Hệ chính quy: {ctdt_chinhquy}"
#                     response += f"<br>- Hệ theo nhu cầu xã hội: {ctdt_theonhucauxh}"
#                     response += f"<br>- Hệ cử tuyển: {ctdt_cutuyen}"
#                     response += f"<br>- Hệ quốc tế tiếng việt: {ctdt_qt_tiengviet}"

#                     #Liên thông
#                     response += f"<br>- Hệ liên thông chính quy: {ctdt_lt_chinhquy}"
#                     response += f"<br>- Hệ liên thông theo nhu cầu xã hội: {ctdt_lt_theonhucauxh}"

#                     # Vừa học vừa làm
#                     response += f"<br>- Hệ vừa học vừa làm:<br>+ Trung cấp đào tạo lên đại học (chính quy): {trungcap_dh_chinhquy}"
#                     response += f"<br>+ Trung cấp đào tạo lên đại học (theo nhu cầu XH): {trungcap_dh_nhu_cau_xh}"
#                     response += f"<br>+ Cao đẳng đào tạo lên đại học (chính quy): {caodang_dh_chinhquy}"
#                     response += f"<br>+ Cao đăng đào tạo lên đại học (theo nhu cầu XH): {caodang_dh_nhucau_xh}"
                    
#                 except KeyError:
#                     response = f"Xin lỗi, tôi không tìm thấy thông tin học phí cho {slot_ten_nganh}."
            
#             # if  'hoi_ctdt' in role and 'ten_nganh' not in role:
#             #     response = response

#         dispatcher.utter_message(text=response)
        
#         return [SlotSet("CTDT_nganh", None)]


  
class actionCTDT(Action):
    def name(self):
        return "action_CTDT_nganh"
    def run(self, dispatcher: CollectingDispatcher,tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        print("ACTION CTĐT ngành")
        entities = tracker.latest_message['entities']
        print(f"\n\nEntity từ input : {entities}")
        infor_ctdt_hoc_phi = constant.get_hocphi_ctdt()
        role = []
        slot_ten_nganh = tracker.get_slot('CTDT_nganh')
        print(f"Slot tên ngành là : {slot_ten_nganh}")
        for entity in entities:
            if entity.get('role') not in role:
                role.append(entity.get('role'))
        print("mảng role: ", role)
        response = "Bạn vui lòng nhập đầy đủ câu hỏi<br> Ví dụ:<br>- Học phí + tên ngành<br>- Chương trình đào tạo + tên ngành"
        found = False
        
        if 'ten_nganh' in role and 'hoi_ctdt' in role:
            response = f"<br>Chương trình đào tạo của {slot_ten_nganh} : "
            print("Chương trình đào tạo của ngành là")
            # Chính Quy 
            #Trúng tuyển chính quy
            if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['chinhquy']:
                print("Chương trình đào tạo chính quy")
                response += "<br>Hệ chính quy :" 
                response += f"<br>- Đào tạo trúng tuyển chính quy : " + infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['chinhquy'][slot_ten_nganh]['ctdt'] 
                found = True
            #đào tạo theo nhu cầu xã hội
            if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['nhu_cau_xh']:
                response += "<br>- Đào tạo theo nhu cầu xã hội (theo đặt hàng ) : " + infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['nhu_cau_xh'][slot_ten_nganh]['ctdt'] 
                found = True
            #đào tạo hệ cử tuyển
            if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['cu_tuyen']:
                response += "<br>- Đào tạo hệ cử tuyển : "+ infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['cu_tuyen'][slot_ten_nganh]['ctdt'] 
                found = True
            #đào tạo sinh viên quốc tế chương trình tiếng việt
            if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['qt_tiengviet']:
                response +="<br>- Đào tạo sinh viên quốc tế chương trình tiếng Việt: " + infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['qt_tiengviet'][slot_ten_nganh]['ctdt'] 
                found = True
            # Liên thông
            # Liên thông chính quy
            if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_lienthong']['lt_chinhquy']:
                response += "<br>Hệ liên thông : "
                response += "<br>- Đào tạo liên thông hệ chính quy : " + infor_ctdt_hoc_phi['daihoc']['he_lienthong']['lt_chinhquy'][slot_ten_nganh]['ctdt'] 
                found = True
            #Liên thông theo nhu cầu xã hội
            if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_lienthong']['lt_nhu_cau_xh']:
                response += "<br>- Đào tạo liên thông theo nhu cầu xã hội : " + infor_ctdt_hoc_phi['daihoc']['he_lienthong']['lt_nhu_cau_xh'][slot_ten_nganh]['ctdt'] 
                found = True
            # Vừa học vừa làm
            if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_vuahoc_vualam']['trungcap_dh_chinhquy']:
                response += "<br>Hệ liên thông vừa học vừa làm :"
                response +="<br>- Hệ trung cấp đầo tạo lên đại học (theo chương trình chính quy): " + infor_ctdt_hoc_phi['daihoc']['he_vuahoc_vualam']['trungcap_dh_chinhquy'][slot_ten_nganh]['ctdt'] 
                found = True
            if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_vuahoc_vualam']['trungcap_dh_nhu_cau_xh']:
                response += "<br>- Hệ trung cấp đào tạo lên đại học (theo nhu cầu xã hội): " + infor_ctdt_hoc_phi['daihoc']['he_vuahoc_vualam']['trungcap_dh_nhu_cau_xh'][slot_ten_nganh]['ctdt'] 
                found = True
            if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_vuahoc_vualam']['caodang_dh_chinhquy']:
                response += "<br>- Hệ cao đẳng đào tạo lên đại học (theo chương trình chính quy): " + infor_ctdt_hoc_phi['daihoc']['he_vuahoc_vualam']['caodang_dh_chinhquy'][slot_ten_nganh]['ctdt'] 
                found = True
            if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_vuahoc_vualam']['caodang_dh_nhucau_xh']:
                response += "<br>- Hệ cao đẳng đào tạo lên đại học (theo nhu cầu xã hội): " + infor_ctdt_hoc_phi['daihoc']['he_vuahoc_vualam']['caodang_dh_nhucau_xh'][slot_ten_nganh]['ctdt'] 
                found = True
            if found == False:
                response = f"Không tìm thấy ngành bạn yêu cầu."
        
        dispatcher.utter_message(text=response)
        
        return [SlotSet("CTDT_nganh", None)]