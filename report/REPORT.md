# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…
`class_id`=468 ; `class_name`="cab" ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`="ImageNet-1K"
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
- Record này nói về cả bức ảnh. Vì đây là tác vụ phân loại ảnh, mô hình phân tích toàn bộ ngữ cảnh và các đặc trưng pixel trên diện rộng để đưa ra dự đoán thuộc lớp nào (như cab, minibus, police_van...). 
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
- Danh sách tên lớp đi kèm với taxonomy/checkpoint đã được định nghĩa từ trước (tập nhãn ImageNet-1K đi kèm checkpoint yolo11n-cls.pt). Mô hình không tự nghĩ ra lớp mới mà chỉ tính toán xác suất cho các lớp có sẵn trong taxonomy này.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
- Đảm bảo tính nhất quán: Các bộ dữ liệu hoặc mô hình khác nhau có thể dùng chung class_name nhưng có class_id khác nhau.
- Tránh trùng lặp: taxonomy_name (như ImageNet-1K hoặc COCO-80) giúp xác định rõ không gian nhãn được tham chiếu, giúp annotator đối chiếu chính xác khi tích hợp nhiều nguồn dữ liệu.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
- Ưu tiên gán nhãn cho vật thể chiếm diện tích lớn nhất hoặc nằm ở trung tâm bức ảnh.
- Ví dụ nếu ảnh chụp đường phố có nhiều xe cộ, có thể quy định gán nhãn là cảnh đường xá (streetcar, cab) thay vì gán nhãn một người đi bộ nhỏ xuất hiện trong góc.
- Hướng dẫn xử lý khi tỷ lệ phân bổ các vật thể ngang nhau (ví dụ: hỏi ý kiến Lab Coach hoặc chọn nhãn chung chung nhất).
6. Vì sao không được lấy `score` của model làm ground truth?
- score chỉ thể hiện mức độ tin cậy của mô hình dựa trên những gì nó đã được học, không phải ground truth.
- Mô hình có thể dự đoán sai với độ tin cậy rất cao hoặc dự đoán đúng nhưng độ tin cậy thấp. Ground truth phải là nhãn chuẩn xác tuyệt đối do người gán nhãn gán và kiểm duyệt dựa theo Guideline thực tế.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…
`class_name`="person" ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
Góc trên-trái (385.33, 69.24), góc dưới-phải (498.92, 348.92), box rộng 113.58 px, cao 279.68 px
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
- Threshold=0.20 có 17 prediction
- Threshold=0.60 có 6 prediction
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
- Lọc bỏ hộp nhiễu: Kiểm tra và xóa bỏ các hộp dự đoán sai đối tượng hoặc các hộp rác do mô hình tự vẽ ra khi độ tự tin thấp.
- Hợp nhất hộp trùng lặp: Xử lý các hộp bị trùng lặp hoặc chồng chéo lên cùng một vật thể mà mô hình không tự triệt tiêu được.
- Chỉnh sửa biên hộp: Do ở ngưỡng thấp, độ chính xác của tọa độ hộp thường bị lệch, reviewer phải căn chỉnh lại các cạnh của hộp cho khít với vật thể hơn.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
- "Hộp gán nhãn phải là hình chữ nhật nhỏ nhất bao bọc hoàn toàn phần nhìn thấy được của vật thể. Không được để thừa khoảng trống quá nhiều ở các cạnh biên, và tuyệt đối không được cắt lẹm vào phần pixel của vật thể."
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
- Trên ảnh kitchen, các chiếc nồi/xoong treo trên giá phía trên bị che khuất một phần lẫn nhau.
- Phương án xử lý: Nên viết thành quy tắc rõ ràng trong Guideline thay vì ca nào cũng đưa lên hỏi. Vì hiện tượng vật bị che khuất hoặc bị cắt mép là những trường hợp phổ biến trong thực tế. Nếu mỗi lần gặp đều hỏi Lab Coach sẽ gây tắc nghẽn quy trình. 
- Guideline cần quy định sẵn: Ví dụ: "Nếu vật thể bị che khuất dưới một mức độ nào đó, vẫn phải vẽ hộp bao quanh phần nhìn thấy được. Nếu bị che khuất trên mức đó hoặc không thể nhận diện được bằng mắt thường, bỏ qua không gán nhãn." Chỉ khi gặp ca cực kỳ mơ hồ không thể tự quyết định theo quy tắc này thì mới đưa lên hỏi Lab Coach (escalation).

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…
`instance_id`="kitchen-001" ; `class_name`="person" ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0],...
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
- Polygon (đa giác) cung cấp thông tin chi tiết về biên dạng thực tế và hình học cụ thể của vật thể ở cấp độ pixel.
- Khác với Box chỉ là khung hình chữ nhật và chứa nhiều pixel nền, Polygon giúp phân định chính xác những vùng pixel nào thuộc về vật thể và vùng nào không, phản ánh đúng các đường cong, góc cạnh, diện tích thực tế bị che khuất hoặc rỗng bên trong.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
- instance_id (như kitchen-001) dùng để định danh và phân biệt từng cá thể riêng biệt trong một bức ảnh duy nhất.
- Không phải class ID: Class ID là mã của lớp đối tượng (ví dụ: person là ID 0), trong khi instance_id gán riêng cho từng đối tượng khác nhau trong ảnh (kitchen-001 là person, kitchen-002 là bowl).
- Không phải tracking ID: Tracking ID dùng để khớp một đối tượng xuyên suốt các khung hình của một video. Còn instance_id ở đây chỉ có giá trị tĩnh, độc lập trong phạm vi duy nhất một bức ảnh tĩnh đang xét.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
- "Đường biên đa giác phải ôm sát theo đường viền thực tế của vật thể. Khoảng cách giữa các điểm neo (vertices) không được vượt quá ... pixel ở các đoạn cong mềm mại, và phải đặt điểm neo chính xác tại các góc bẻ đột ngột để đảm bảo không bỏ sót hoặc lấy dư pixel nền quá ... px."
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
- Trong ảnh kitchen, khu vực các chiếc bát xếp cạnh nhau trên bàn gỗ.
- Phương án xử lý: Nên kết hợp định nghĩa rõ trong Guideline các nguyên lý cơ bản, nhưng cần cơ chế Escalation cho các ca cực kỳ mờ hoặc dính liền.
- Vì với các trường hợp chạm nhau hoặc che khuất nhẹ, Guideline cần quy định sẵn nguyên tắc phân tách: "Vẽ biên hoàn chỉnh cho vật nằm đè lên trước, sau đó vẽ biên phần nhìn thấy được cho vật bị che phía sau.". Tuy nhiên, với các khu vực biên quá mờ do thiếu sáng hoặc các vật bị hòa lẫn màu sắc với nhau không thể phân biệt ranh giới bằng mắt thường, người gán nhãn phải đưa lên hỏi Lab Coach (escalation) để tránh việc tự đoán mò làm sai lệch phân bố nhãn chuẩn xác (ground truth).

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn duy nhất kèm theo định danh taxonomy cho toàn bộ ảnh (ví dụ: class_name = "cab"). | Khó quyết định ngữ cảnh: Ảnh traffic có rất nhiều loại xe hỗn hợp (bus, car, minibus). Việc ép cả ảnh vào 1 nhãn duy nhất (cab - score 0.51) bỏ qua toàn bộ các đối tượng quan trọng khác. | Đọc guideline, xem toàn bộ ảnh và chọn ra một lớp đặc trưng nhất theo quy tắc đã thống nhất (ví dụ: đối tượng chiếm diện tích lớn nhất). | Đánh giá xem nhãn được chọn có đúng với quy tắc trong guideline hay không; kiểm tra các ca nhầm lẫn giữa các lớp tương đồng. |
| Phát hiện vật thể | Các hình chữ nhật bao quanh từng vật thể, lưu dưới dạng pixel: [x_min, y_min, x_max, y_max] kèm tên lớp. | Bỏ sót & Che khuất: Trên ảnh kitchen, các vật thể bị che khuất một phần (như những chiếc nồi treo chồng chéo ở góc trên bên trái) bị mô hình bỏ sót hoặc gán sai nhãn; ranh giới hộp dễ bị thừa/thiếu pixel. | Vẽ hộp hình chữ nhật bao khít tối đa phần nhìn thấy được của đối tượng, gán nhãn chính xác từ danh sách lớp. | Kiểm tra độ khít của các cạnh hộp (không thừa/thiếu quá mức); rà soát các lỗi bỏ sót vật thể nhỏ hoặc gán sai nhãn. |
| Instance segmentation | Tập hợp các điểm tọa độ đa giác [[x1, y1], [x2, y2], ...] bao quanh chi tiết ranh giới của từng đối tượng độc lập kèm instance_id | Biên mờ & Dính nhau: Trên ảnh kitchen, khu vực chiếc bàn gỗ lớn có các bát dính sát nhau làm mờ ranh giới, mô hình gán nhãn đè lấn lên nhau rất khó phân tách chi tiết. | Vẽ polyon bám sát từng pixel đường viền thực tế của đối tượng, phân biệt rõ các đối tượng cạnh nhau bằng cách đặt instance_id riêng biệt. | Soi kỹ polygon xem có bị lẹm hoặc bao quá rộng so với vật thể không, kiểm tra ranh giới phân tách giữa các đối tượng xếp chồng/chạm nhau. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Tuyệt đối không chia sẻ công khai bất kỳ dữ liệu nội bộ, ảnh cá nhân hoặc thông tin nhạy cảm nào ngoài 3 bức ảnh COCO công khai đã được cung cấp sẵn trong bài thực hành.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Em sẽ lập tức dừng mọi hoạt động xử lý dữ liệu đó, giữ nguyên trạng hệ thống và báo cáo ngay lập tức cho Lab Coach để nhận hướng dẫn xử lý, tuyệt đối không tự ý sao chép hoặc phát tán.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
