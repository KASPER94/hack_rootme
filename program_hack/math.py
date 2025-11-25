import requests
import re
import sys

def solve():
    # Configuration
    base_url = "http://challenge01.root-me.org/programmation/ch1/"
    submit_url_endpoint = "ep1_v.php"
    
    # Use a session to persist cookies (PHPSESSID)
    s = requests.Session()
    # Add headers to mimic a browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        # 1. Get the challenge page
        print(f"[*] Fetching {base_url}...")
        r = s.get(base_url, headers=headers)
        if r.status_code != 200:
            print(f"[-] Error fetching page: {r.status_code}")
            return

        html = r.text
        
        # Cleaning HTML tags for easier regex (optional, but helps if structure is messy)
        # Or just parse raw HTML with loose regex.
        # Expected format roughly: Un+1 = [ A + Un ] +/- [ n * B ] ... U0 = ... Find U_target
        
        # Extract A (the constant added to Un)
        # Matches: [ -13 + Un ] or [ 50 + U ] -> Handles Un and U<sub>n</sub>
        match_a = re.search(r"\[\s*([+-]?\d+)\s*\+\s*U", html)
        
        # Extract Operator between brackets (+ or -)
        # Matches: ] + [ or ] - [
        match_op = re.search(r"\]\s*([+-])\s*\[", html)
        
        # Extract B (the coefficient for n)
        # Matches: [ n * 4 ] or [ n * -31 ]
        match_b = re.search(r"n\s*\*\s*([+-]?\d+)\s*\]", html)
        
        # Extract U0
        # Matches: U0 = 939 or U<sub>0</sub> = 939
        match_u0 = re.search(r"U(?:<sub>)?0(?:</sub>)?\s*=\s*([+-]?\d+)", html)
        
        # Extract Target Index
        # Matches: find U759769 or find U<sub>759769</sub>
        match_target = re.search(r"find\s*U(?:<sub>)?(\d+)(?:</sub>)?", html)

        if not (match_a and match_op and match_b and match_u0 and match_target):
            print("[-] Failed to parse parameters from HTML.")
            print("Debug HTML fragment:")
            print(html[:500]) # Print first 500 chars to debug
            return

        a = int(match_a.group(1))
        op = match_op.group(1)
        b_raw = int(match_b.group(1))
        u0 = int(match_u0.group(1))
        n_target = int(match_target.group(1))

        # Adjust B based on the operator
        # If formula is ... - [ n * B ], then effective B term is -B
        if op == '-':
            b = -b_raw
        else:
            b = b_raw

        print(f"[*] Parsed: Un+1 = [{a} + Un] + [n * {b}]")
        print(f"[*] U0 = {u0}")
        print(f"[*] Target: U_{n_target}")

        # 2. Calculate U_target
        # Recurrence: U_{n+1} = U_n + A + B*n
        # Summation form: U_n = U_0 + n*A + B * Sum(0 to n-1 of k)
        # Sum(0 to k) = k(k+1)/2 -> Sum(0 to n-1) = (n-1)n/2
        
        term_const = n_target * a
        term_n = b * (n_target * (n_target - 1) // 2)
        result = u0 + term_const + term_n

        print(f"[*] Calculated Result: {result}")

        # 3. Send result
        # URL format: .../ep1_v.php?result=...
        submit_url = f"{base_url}{submit_url_endpoint}?result={result}"
        
        print(f"[*] Sending to {submit_url}...")
        r_submit = s.get(submit_url, headers=headers)
        
        # 4. Check response
        if "congratulations" in r_submit.text.lower() or "password" in r_submit.text.lower() or "html" not in r_submit.text:
             print("[+] SUCCESS! Server response:")
             print(r_submit.text.strip())
        else:
             print("[-] Failed? Response:")
             print(r_submit.text[:200]) # Print beginning of response

    except Exception as e:
        print(f"[-] Exception: {e}")

if __name__ == "__main__":
    solve()

