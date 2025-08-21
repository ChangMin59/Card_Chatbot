import json
import time
import os
import random
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from webdriver_manager.chrome import ChromeDriverManager

# Selenium WebDriver 설정
options = Options()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

# 크롤링 데이터 저장 경로 설정
output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "crawled")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "card_data.json")

# 카드 페이지 접속
card_list_url = "https://www.card-gorilla.com/card?cate=CRD"
driver.get(card_list_url)
wait = WebDriverWait(driver, 10)

card_db = {}

while True:
    try:
        detail_buttons = driver.find_elements(By.CSS_SELECTOR, "a.b_view")
        start_idx = len(card_db)

        if not detail_buttons:
            print("[INFO] 더 이상 크롤링할 카드가 없습니다.")
            break

        for idx in range(start_idx, start_idx + 10):
            detail_buttons = driver.find_elements(By.CSS_SELECTOR, "a.b_view")
            if idx >= len(detail_buttons):
                break

            try:
                print(f"[INFO] {len(card_db)+1}번째 카드 크롤링 중...")
                driver.execute_script("arguments[0].click();", detail_buttons[idx])
                time.sleep(random.uniform(1, 2))

                # 팝업 닫기
                try:
                    popup_close = WebDriverWait(driver, 2).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, "button.swal2-close"))
                    )
                    popup_close.click()
                    time.sleep(0.5)
                except TimeoutException:
                    pass

                try:
                    close_button = driver.find_element(By.CSS_SELECTOR, "div.btn_md_close")
                    close_button.click()
                    time.sleep(0.5)
                except NoSuchElementException:
                    pass

                # 광고창 감지 닫기
                all_windows = driver.window_handles
                if len(all_windows) > 1:
                    for w in all_windows:
                        if w != driver.current_window_handle:
                            driver.switch_to.window(w)
                            driver.close()
                    driver.switch_to.window(driver.window_handles[0])

                # 발급 중단 카드 스킵
                discontinued = driver.find_elements(By.XPATH, "//b[contains(text(), '신규발급이 중단된 카드입니다.')]")
                if discontinued:
                    print("신규발급 중단 카드 - 스킵")
                    driver.back()
                    WebDriverWait(driver, 7).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "a.b_view"))
                    )
                    time.sleep(random.uniform(1, 2))
                    continue

                # 카드 정보 추출
                try:
                    card_name = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "strong.card"))).text.strip()
                except TimeoutException:
                    card_name = "정보 없음"

                try:
                    card_company = driver.find_element(By.CSS_SELECTOR, "p.brand").text.strip()
                except NoSuchElementException:
                    card_company = "정보 없음"

                # 혜택 추출
                keyword_list = []
                benefit_desc = {}
                try:
                    benefit_blocks = driver.find_elements(By.CSS_SELECTOR, "dl")
                    for block in benefit_blocks[:-1]:
                        try:
                            keyword = block.find_element(By.CSS_SELECTOR, "p.txt1").text.strip()
                            desc = block.find_element(By.CSS_SELECTOR, "i").text.strip()

                            if keyword:
                                # 키워드 중복 제거용 (리스트에는 append 전에 검사)
                                if keyword not in keyword_list:
                                    keyword_list.append(keyword)

                                # 혜택 설명 병합 처리
                                if keyword in benefit_desc:
                                    if desc not in benefit_desc[keyword]:
                                        benefit_desc[keyword] += ", " + desc
                                else:
                                    benefit_desc[keyword] = desc

                        except NoSuchElementException:
                            continue
                except NoSuchElementException:
                    pass

                card_db[card_name] = {
                    "카드 회사": card_company,
                    "혜택 키워드": keyword_list,
                    "혜택 설명": benefit_desc
                }

                print(f"[SUCCESS] {card_name} 저장 완료!")

                # 실시간 저장
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(card_db, f, ensure_ascii=False, indent=4)

                # 목록 페이지 복귀
                driver.back()
                WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "a.b_view"))
                )
                time.sleep(random.uniform(1, 2))

            except Exception as e:
                print(f"[ERROR] 카드 상세 크롤링 실패: {e}")

    except Exception as e:
        print(f"[ERROR] 카드 목록 처리 중 오류: {e}")
        break

    # 더보기 버튼 클릭
    try:
        more_button = driver.find_element(By.CSS_SELECTOR, "a.lst_more")
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", more_button)
        time.sleep(0.5)
        driver.execute_script("arguments[0].click();", more_button)
        print("[INFO] '더보기' 클릭됨. 추가 카드 로드 중...")
        time.sleep(random.uniform(1, 2))
    except NoSuchElementException:
        print("[INFO] 더 이상 '더보기' 버튼이 없습니다. 종료합니다.")
        break
    except Exception as e:
        print(f"[WARNING] '더보기' 클릭 실패: {e}")
        break

# 크롤링 완료 출력
print(f"\n총 {len(card_db)}개의 카드 정보를 수집했습니다.")
print(f"결과 저장 위치: {output_file}")
driver.quit()
